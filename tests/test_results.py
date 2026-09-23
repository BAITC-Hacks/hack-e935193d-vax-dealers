"""Проверка содержимого итоговых файлов и автономной панели."""

import copy
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wind_agent as agent
from dashboard import build_dashboard
from result_checks import (
    HASHED_FILES,
    RESULT_FILES,
    check_results,
    inspect_snapshot,
    load_snapshot,
)
from test_regressions import write_model


def change_table(content, name, operation):
    table = pd.read_csv(io.BytesIO(content[name]))
    modified = operation(table)
    content[name] = (
        (table if modified is None else modified).to_csv(index=False).encode()
    )


def resign(content):
    """Use fresh hashes to test semantic checks independently of hash checking."""
    summary = json.loads(content["run_summary.json"])
    summary["artifact_sha256"] = {
        name: hashlib.sha256(content[name]).hexdigest() for name in HASHED_FILES
    }
    content["run_summary.json"] = json.dumps(summary).encode()


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root):
                agent.replay(offline=True)
            cls.original, _ = load_snapshot(root)

    def setUp(self):
        self.content = copy.deepcopy(self.original)

    def test_complete_results_are_valid_but_assumptions_remain_unconfirmed(self):
        result = inspect_snapshot(self.content)
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(result["assumptions_confirmed"])
        self.assertEqual(result["counts"]["february_rows"], 1344)

    def test_changed_bytes_fail_integrity_check(self):
        self.content["forecasts_48h.csv"] += b"\n"
        result = inspect_snapshot(self.content)
        self.assertFalse(result["valid"])
        self.assertTrue(any("Контрольная сумма" in e for e in result["errors"]))

    def test_missing_hour_is_detected_even_with_updated_hashes(self):
        change_table(self.content, "forecasts_48h.csv", lambda frame: frame.iloc[1:])
        resign(self.content)
        self.assertFalse(inspect_snapshot(self.content)["valid"])

    def test_wrong_horizon_is_detected_even_with_updated_hashes(self):
        change_table(
            self.content,
            "forecasts_48h.csv",
            lambda frame: frame.__setitem__("lead_hours", frame.lead_hours + 1),
        )
        resign(self.content)
        result = inspect_snapshot(self.content)
        self.assertTrue(
            any("Горизонт" in e or "горизонт" in e for e in result["errors"])
        )

    def test_submission_must_match_day_ahead_forecasts(self):
        change_table(
            self.content,
            "submission_february.csv",
            lambda frame: frame.__setitem__("prediction", 0.45),
        )
        resign(self.content)
        result = inspect_snapshot(self.content)
        self.assertTrue(any("подача не совпадает" in e for e in result["errors"]))

    def test_station_must_match_turbine_predictions(self):
        change_table(
            self.content,
            "station_february.csv",
            lambda frame: frame.__setitem__("turbine_1_normalized", 0.99),
        )
        resign(self.content)
        result = inspect_snapshot(self.content)
        self.assertTrue(any("Таблица станции" in e for e in result["errors"]))

    def test_interval_must_contain_prediction(self):
        change_table(
            self.content,
            "forecasts_48h.csv",
            lambda frame: frame.__setitem__("lower90", 0.9),
        )
        resign(self.content)
        result = inspect_snapshot(self.content)
        self.assertTrue(any("Диапазон неопределённости" in e for e in result["errors"]))

    def test_malformed_summary_returns_failure_not_success(self):
        self.content["run_summary.json"] = b"not json"
        self.assertFalse(inspect_snapshot(self.content)["valid"])

    def test_archive_does_not_mix_with_partial_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            with zipfile.ZipFile(root / "results/forecast_results.zip", "w") as z:
                for name, value in self.content.items():
                    z.writestr("results/" + name, value)
            (root / "results/run_summary.json").write_text("{new partial run")
            report = check_results(root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["source"], "archive")

    def test_missing_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            with zipfile.ZipFile(root / "results/forecast_results.zip", "w") as z:
                z.writestr("results/run_summary.json", self.content["run_summary.json"])
            self.assertFalse(check_results(root)["valid"])

    def test_dashboard_is_self_contained_and_json_cannot_close_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            for name, value in self.content.items():
                (root / "results" / name).write_bytes(value)
            summary = json.loads((root / "results/run_summary.json").read_text())
            summary["note"] = "</script><script>bad()</script>"
            (root / "results/run_summary.json").write_text(json.dumps(summary))
            shutil.copytree(agent.ROOT / "ui", root / "ui")
            html = build_dashboard(root).read_text()
            self.assertNotIn("</script><script>bad()", html)
            self.assertNotIn("__FORECAST_DATA__", html)
            self.assertNotIn("__DASHBOARD_SCRIPT__", html)
            self.assertNotIn("__DASHBOARD_STYLE__", html)
            script = re.search(
                r'<script\s+id="forecast-data"\s+type="application/json">\s*(.*?)\s*</script>',
                html,
                re.S,
            )
            self.assertIsNotNone(script)
            payload = json.loads(script.group(1))
            self.assertEqual(len(payload["rows"]), 2784)
            self.assertEqual(payload["summary"]["note"], summary["note"])

            class Assets(HTMLParser):
                external = []

                def handle_starttag(self, tag, attrs):
                    attrs = dict(attrs)
                    if tag in ("script", "img", "iframe", "link"):
                        url = attrs.get("src") or attrs.get("href")
                        if url:
                            self.external.append(url)

            parser = Assets()
            parser.feed(html)
            self.assertEqual(parser.external, [])

    def test_corrupted_output_is_recomputed_by_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root)
            with patch.object(agent, "ROOT", root):
                agent.replay("2026-02-10", "2026-02-10", offline=True)
                path = root / "results/cycles/2026-02-10/forecasts_48h.csv"
                path.write_text("corrupted output")
                with patch.object(
                    agent, "predict_with_fallback", wraps=agent.predict_with_fallback
                ) as predict:
                    agent.replay(
                        "2026-02-10", "2026-02-10", offline=True, skip_unchanged=True
                    )
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(len(pd.read_csv(path)), 96)


if __name__ == "__main__":
    unittest.main()
