import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wind_agent as agent


class AgentTests(unittest.TestCase):
    def test_future_weather_guard_all_horizons(self):
        plan = agent.schedule("2026-01-31", "2026-02-28")
        self.assertEqual(len(plan), 2784)
        self.assertTrue((plan.availability_bound_utc <= plan.issue_utc).all())
        self.assertEqual(
            list(agent.safe_offset([1, 12, 13, 36, 37, 48])), [1, 1, 2, 2, 3, 3]
        )
        self.assertFalse(
            plan.duplicated(["issue_utc", "valid_utc", "turbine_id"]).any()
        )

    def test_february_has_one_day_ahead_prediction_per_hour(self):
        plan = agent.schedule("2026-01-31", "2026-02-28")
        feb = plan[(plan.lead_hours <= 24) & (plan.valid_local < "2026-03-01")]
        self.assertEqual(len(feb), 1344)
        self.assertEqual(feb.groupby("turbine_id").size().to_dict(), {1: 672, 2: 672})
        self.assertFalse(feb.duplicated(["valid_local", "turbine_id"]).any())

    def test_hourly_target_requires_all_six_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tid in [1, 2]:
                times = pd.date_range("2025-01-01", periods=11, freq="10min")
                pd.DataFrame(
                    {
                        "id": range(11),
                        "time": times,
                        "wind": 5.0,
                        "power": 0.3,
                        "temp": 10.0,
                    }
                ).to_csv(root / f"turbine {tid}.csv", index=False)
            with patch.object(agent, "ROOT", root):
                frame = agent.prepare(root)
            self.assertEqual(frame.power.notna().sum(), 2)
            self.assertEqual(frame.power.isna().sum(), 2)
            self.assertTrue(np.allclose(frame.power.dropna(), 0.3))

    def test_duplicate_scada_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [[1, "2025-01-01", 4, 0.2, 10], [2, "2025-01-01", 4, 0.2, 10]],
                columns=["id", "time", "wind", "power", "temp"],
            ).to_csv(root / "turbine 1.csv", index=False)
            with patch.object(agent, "ROOT", root), self.assertRaisesRegex(
                ValueError, "Duplicate"
            ):
                agent.prepare(root)

    def test_no_actual_power_or_scada_wind_in_features(self):
        frame = agent.schedule("2026-01-31", "2026-01-31")
        frame["wind"] = 5.0
        frame["temp"] = 10.0
        frame["direction"] = 90.0
        frame["power"] = 1.0
        first = agent.features(frame)
        frame["power"] = 0.0
        pd.testing.assert_frame_equal(first, agent.features(frame))
        self.assertNotIn("power", first.columns)

    def test_missing_weather_is_preserved_for_fallback(self):
        plan = agent.schedule("2026-01-31", "2026-01-31")
        weather = (
            plan[["valid_utc", "valid_local", "turbine_id", "offset_days"]]
            .iloc[:1]
            .copy()
        )
        weather["wind"] = 5.0
        weather["temp"] = 10.0
        weather["direction"] = 90.0
        merged = agent.forecast_frame(plan, weather)
        self.assertEqual(len(merged), 96)
        self.assertEqual(merged.wind.isna().sum(), 95)

    def test_agent_publishes_explicit_fallback_when_weather_missing(self):
        weather = agent.schedule("2026-01-31", "2026-01-31")[
            ["valid_utc", "valid_local", "turbine_id", "offset_days"]
        ].copy()
        for col in ["wind", "temp", "direction"]:
            weather[col] = np.nan
        bundle = {
            "model": None,
            "means": {1: 0.3, 2: 0.4},
            "radii": {"1_1": 0.2, "1_2": 0.2, "2_1": 0.2, "2_2": 0.2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle["ready_local"] = "2026-01-31 23:00"

            def load_weather(**kwargs):
                agent.dump(root / "reports/weather_manifest.json", [])
                agent.dump(root / "reports/weather_cache_problems.json", [])
                return weather

            with patch.object(agent, "ROOT", root), patch.object(
                agent, "load_weather", side_effect=load_weather
            ), patch.object(agent, "load_model", return_value=bundle), patch.object(
                agent, "digest", return_value="test-model"
            ):
                agent.replay("2026-01-31", "2026-01-31", offline=True)
            result = pd.read_csv(root / "results/cycles/2026-01-31/forecasts_48h.csv")
            self.assertEqual(len(result), 96)
            self.assertTrue((result.status == "fallback_climatology").all())
            self.assertTrue((result.lower90 == 0).all())
            self.assertTrue((result.upper90 == 1).all())
            self.assertFalse((root / "results/forecasts_48h.csv").exists())


if __name__ == "__main__":
    unittest.main()
