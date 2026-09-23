"""Проверки ошибок, найденных при ревизии проекта.

Тесты работают без интернета. Маленькие синтетические данные используются
только для проверки поведения программы, не для заявлений о качестве модели.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wind_agent as agent
import weather_archive as weather


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 0.5)


def write_model(root, **changes):
    bundle = dict(
        model=ConstantModel(),
        means={1: 0.3, 2: 0.4},
        radii={"1_1": 0.2, "1_2": 0.2, "2_1": 0.2, "2_2": 0.2},
        version=agent.MODEL_VERSION,
        features=agent.FEATURES,
        settings_signature=agent.settings_signature(),
        ready_local="2026-01-31 23:00",
    )
    bundle.update(changes)
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, root / "models/forecast.joblib")
    # Source files participate in the forecast fingerprint.
    for name in ["wind_agent.py", "weather_archive.py"]:
        (root / name).write_text("test source", encoding="utf-8")


def payload(start="2026-02-01", end="2026-02-28", tid=1):
    times = pd.date_range(
        start, pd.Timestamp(end) + pd.Timedelta(days=1), freq="h", inclusive="left"
    )
    data = dict(
        utc_offset_seconds=0,
        hourly={"time": times.strftime("%Y-%m-%dT%H:%M").tolist()},
        hourly_units={},
        _provenance=dict(
            lat=agent.COORDS[tid][0],
            lon=agent.COORDS[tid][1],
            retrieved_at="2026-09-23T12:00:00Z",
        ),
    )
    for days in (1, 2, 3):
        for variable, unit, value in zip(
            weather.VARIABLES, weather.UNITS, (6.0, 90.0, 10.0)
        ):
            name = f"{variable}_previous_day{days}"
            data["hourly"][name] = [value] * len(times)
            data["hourly_units"][name] = unit
    return data


class RegressionTests(unittest.TestCase):
    def test_unfinished_hour_is_excluded_from_training_and_selection(self):
        frame = pd.DataFrame(
            {
                "valid_local": pd.to_datetime(
                    ["2025-12-31 22:00", "2025-12-31 23:00", "2026-01-01 00:00"]
                )
            }
        )
        self.assertEqual(
            agent.completed_before(frame, agent.POINT_READY_LOCAL).tolist(),
            [True, False, False],
        )

    def test_unfinished_calibration_hour_is_excluded(self):
        frame = pd.DataFrame(
            {"valid_local": pd.to_datetime(["2026-01-31 22:00", "2026-01-31 23:00"])}
        )
        self.assertEqual(
            agent.completed_before(frame, agent.CALIBRATION_READY_LOCAL).tolist(),
            [True, False],
        )

    def test_reversed_or_timed_issue_dates_fail_clearly(self):
        for first, last in [
            ("2026-02-03", "2026-02-01"),
            ("2026-02-01 12:00", "2026-02-02"),
        ]:
            with self.assertRaises(ValueError):
                agent.schedule(first, last)

    def test_partial_range_does_not_overwrite_full_replay(self):
        self.assertNotEqual(
            agent.results_directory("2026-02-01", "2026-02-02"), agent.ROOT / "results"
        )
        self.assertEqual(
            agent.results_directory(*agent.FULL_REPLAY), agent.ROOT / "results"
        )

    def test_same_month_has_same_cache_key_for_every_day(self):
        self.assertEqual(
            list(weather.month_ranges("2026-02-10", "2026-02-11")),
            list(weather.month_ranges("2026-02-01", "2026-02-28")),
        )

    def test_all_weather_units_are_checked(self):
        for name in [
            "wind_speed_80m_previous_day3",
            "temperature_2m_previous_day2",
            "wind_direction_80m_previous_day1",
        ]:
            data = payload()
            data["hourly_units"][name] = "wrong"
            with self.assertRaisesRegex(ValueError, "units"):
                weather.validate_payload(data, "2026-02-01", "2026-02-28")

    def test_truncated_duplicate_and_non_utc_weather_are_rejected(self):
        data = payload()
        data["hourly"]["time"][1] = data["hourly"]["time"][0]
        with self.assertRaisesRegex(ValueError, "timestamps"):
            weather.validate_payload(data, "2026-02-01", "2026-02-28")
        data = payload()
        data["hourly"]["temperature_2m_previous_day3"].pop()
        with self.assertRaisesRegex(ValueError, "truncated"):
            weather.validate_payload(data, "2026-02-01", "2026-02-28")
        data = payload()
        data["utc_offset_seconds"] = 18000
        with self.assertRaisesRegex(ValueError, "UTC"):
            weather.validate_payload(data, "2026-02-01", "2026-02-28")

    def test_nonfinite_and_out_of_range_weather_uses_fallback(self):
        frame = agent.schedule("2026-02-01", "2026-02-01").iloc[:4].copy()
        frame["wind"] = [5.0, np.inf, 6.0, 4.0]
        frame["temp"] = [10.0, 10.0, -100.0, 10.0]
        frame["direction"] = [90.0, 90.0, 90.0, 400.0]
        result, good = agent.predict_with_fallback(
            frame, ConstantModel(), {1: 0.3, 2: 0.4}
        )
        self.assertEqual(good.tolist(), [True, False, False, False])
        self.assertEqual(result.tolist(), [0.5, 0.4, 0.3, 0.4])

    def test_empty_cache_produces_full_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root):
                summary = agent.replay(offline=True)
            self.assertEqual(summary["fallback_rows"], 2784)
            self.assertEqual(summary["february_rows"], 1344)
            result = pd.read_csv(root / "results/submission_february.csv")
            self.assertTrue(result.prediction.notna().all())
            self.assertTrue(result.lower90.eq(0).all())
            self.assertTrue(result.upper90.eq(1).all())

    def test_corrupt_cache_does_not_prevent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache").mkdir()
            (root / "cache/gfs_t1_2026-02-01_2026-02-28.json").write_text("{broken")
            frame, manifest, problems = weather.read_archive(
                root, agent.COORDS, tolerant=True
            )
            self.assertTrue(frame.empty)
            self.assertEqual(len(problems), 1)
            with self.assertRaises(ValueError):
                weather.read_archive(root, agent.COORDS)

    def test_failed_refresh_preserves_previous_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "cache/gfs_t1_2026-02-01_2026-02-28.json"
            weather.atomic_json(path, payload())
            before = path.read_bytes()
            with (
                patch.object(
                    weather.urllib.request, "urlopen", side_effect=OSError("offline")
                ),
                patch.object(weather.time, "sleep"),
            ):
                with self.assertRaises(OSError):
                    weather.fetch_month(
                        root, agent.COORDS, 1, "2026-02-01", "2026-02-28", True
                    )
            self.assertEqual(before, path.read_bytes())

    def test_auto_fetch_is_attempted_before_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root), patch.object(
                agent, "fetch_month", side_effect=OSError("offline")
            ) as fetch:
                result = agent.replay("2026-02-10", "2026-02-10")
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(result["fallback_rows"], 96)

    def test_old_model_or_changed_settings_require_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for change in [dict(version=1), dict(settings_signature="different")]:
                write_model(root, **change)
                with patch.object(agent, "ROOT", root), self.assertRaises(ValueError):
                    agent.load_model()

    def test_replay_before_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root), self.assertRaisesRegex(
                ValueError, "not available"
            ):
                agent.replay("2026-01-30", "2026-01-30", offline=True)
            self.assertFalse((root / "results").exists())

    def test_strict_availability_does_not_claim_unverified_success(self):
        with self.assertRaisesRegex(ValueError, "publication"):
            agent.replay(require_verified_availability=True)

    def test_watch_skips_prediction_when_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root):
                agent.replay("2026-02-10", "2026-02-10", offline=True)
                result_path = root / "results/cycles/2026-02-10/forecasts_48h.csv"
                before = result_path.stat().st_mtime_ns
                with patch.object(
                    agent,
                    "predict_with_fallback",
                    side_effect=AssertionError("recomputed"),
                ):
                    agent.replay(
                        "2026-02-10", "2026-02-10", offline=True, skip_unchanged=True
                    )
                self.assertEqual(before, result_path.stat().st_mtime_ns)
            last = json.loads(
                (root / "results/agent_events.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(last["state"], "UNCHANGED")

    def test_all_invalid_measurements_are_preserved_as_missing_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tid in (1, 2):
                pd.DataFrame(
                    dict(
                        id=range(6),
                        time=pd.date_range("2025-01-01", periods=6, freq="10min"),
                        wind=5.0,
                        power=2.0,
                        temp=10.0,
                    )
                ).to_csv(root / f"turbine {tid}.csv", index=False)
            with patch.object(agent, "ROOT", root):
                result = agent.prepare(root)
            self.assertEqual(len(result), 2)
            self.assertTrue(result.power.isna().all())
            self.assertTrue(result.n_samples.eq(0).all())


if __name__ == "__main__":
    unittest.main()
