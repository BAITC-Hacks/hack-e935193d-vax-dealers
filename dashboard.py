"""Собирает автономную HTML-панель из проверенных прогнозов, без веб-сервера."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from result_checks import inspect_snapshot, load_snapshot


def build_dashboard(root):
    root = Path(root)
    content, source = load_snapshot(root)
    check = inspect_snapshot(content)
    if not check["valid"]:
        raise ValueError("Панель не создана: " + "; ".join(check["errors"]))
    summary = json.loads(content["run_summary.json"])
    frame = pd.read_csv(io.BytesIO(content["forecasts_48h.csv"]))
    issue_local = pd.to_datetime(frame.issue_utc, utc=True) + pd.Timedelta(
        hours=summary["utc_offset_hours"]
    )
    frame["issue_date"] = issue_local.dt.strftime("%Y-%m-%d")
    columns = [
        "issue_date",
        "valid_local",
        "turbine_id",
        "lead_hours",
        "prediction",
        "lower90",
        "upper90",
        "wind",
        "temp",
        "status",
    ]
    metrics, audit = [], {}
    card_path = root / "reports/model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("model_sha256") == summary["model_sha256"]:
            metrics_path = root / "reports/january_metrics.csv"
            if metrics_path.exists():
                metrics = json.loads(
                    pd.read_csv(metrics_path).to_json(
                        orient="records", double_precision=12
                    )
                )
    audit_path = root / "reports/data_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    data = dict(
        rows=json.loads(frame[columns].to_json(orient="records", double_precision=12)),
        summary=summary,
        check=check,
        metrics=metrics,
        audit=audit,
        source=source,
    )
    # Embedded JSON cannot close its script element, even if a source contains HTML.
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = (root / "ui/dashboard.html").read_text(encoding="utf-8")
    replacements = {
        "__FORECAST_DATA__": payload,
        "__DASHBOARD_STYLE__": (root / "ui/dashboard.css").read_text(encoding="utf-8"),
        "__DASHBOARD_SCRIPT__": (root / "ui/forecast_logic.js").read_text(
            encoding="utf-8"
        )
        + "\n"
        + (root / "ui/dashboard.js").read_text(encoding="utf-8"),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"В шаблоне должен быть ровно один маркер {marker}")
        template = template.replace(marker, value)
    output = root / "results/dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(template, encoding="utf-8")
    temporary.replace(output)
    return output
