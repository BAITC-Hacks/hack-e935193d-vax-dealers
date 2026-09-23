"""Vax-Dealers: подготовка данных, обучение и ежедневный прогноз двух турбин."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import zipfile
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import joblib
from weather_archive import (
    atomic_json,
    fetch_month as download_month,
    month_ranges,
    read_archive,
    valid_weather,
)

ROOT = Path(__file__).resolve().parent
COORDS = {1: (43.645150, 78.535604), 2: (43.643198, 78.538828)}
VARIABLES = ["wind_speed_80m", "wind_direction_80m", "temperature_2m"]
FEATURES = [
    "wind",
    "temp",
    "direction_sin",
    "direction_cos",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "offset_days",
    "turbine_id",
]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
if not -12 <= CONFIG["utc_offset_hours"] <= 14:
    raise ValueError("config.json: utc_offset_hours must be between -12 and 14")
if not 0 <= CONFIG["weather_latency_hours"] <= 24:
    raise ValueError("config.json: weather_latency_hours must be between 0 and 24")
OFFSET = pd.Timedelta(hours=CONFIG["utc_offset_hours"])
LATENCY_HOURS = CONFIG["weather_latency_hours"]
MODEL_VERSION = 2
POINT_READY_LOCAL = pd.Timestamp("2025-12-31 23:00")
CALIBRATION_READY_LOCAL = pd.Timestamp("2026-01-31 23:00")
FULL_REPLAY = ("2026-01-31", "2026-02-28")


def settings_signature():
    settings = dict(
        offset_seconds=OFFSET.total_seconds(),
        latency=LATENCY_HOURS,
        coordinates=COORDS,
        variables=VARIABLES,
        features=FEATURES,
        time_assumptions_confirmed=CONFIG["time_assumptions_confirmed"],
    )
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def completed_before(frame, issue_local):
    """A value labelled 22:00 becomes known when the hour ends at 23:00."""
    return frame.valid_local + pd.Timedelta(hours=1) <= pd.Timestamp(issue_local)


def dump(path, data):
    atomic_json(path, data)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_month(tid, start, end, refresh=False):
    return download_month(ROOT, COORDS, tid, start, end, refresh)


def fetch_range(start, end, refresh=False):
    for first, last in month_ranges(start, end):
        for tid in COORDS:
            print("Weather:", fetch_month(tid, first, last, refresh).name, flush=True)


def prepare(data_dir):
    frames, audit = [], {}
    for tid in COORDS:
        matches = list(Path(data_dir).glob(f"*turbine {tid}.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one turbine {tid}.csv, got {matches}")
        path = matches[0]
        raw = pd.read_csv(path)
        if len(raw.columns) != 5:
            raise ValueError("Unexpected SCADA schema")
        columns = list(raw.columns)
        expected = [
            "ID",
            "Статистическое время",
            "Средняя скорость ветра(m/s)",
            "Нормализованная активная мощность",
            "Средняя температура окружающей среды(°C)",
        ]
        if columns not in (expected, ["id", "time", "wind", "power", "temp"]):
            raise ValueError(f"Unexpected SCADA column names/order: {columns}")
        if raw.empty:
            raise ValueError("SCADA file is empty")
        raw.columns = ["id", "time", "wind", "power", "temp"]
        raw["time"] = pd.to_datetime(raw.time, errors="raise")
        if raw.time.isna().any() or raw.time.dt.tz is not None:
            raise ValueError(
                "SCADA timestamps must be non-empty local times without a timezone"
            )
        if raw.time.duplicated().any():
            raise ValueError("Duplicate SCADA timestamps")
        for col in ["power", "wind", "temp"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        bad = (
            ~raw.power.between(0, 1)
            | ~raw.wind.between(0, 80)
            | ~raw.temp.between(-70, 70)
        )
        off_grid = raw.time != raw.time.dt.floor("10min")
        bad |= off_grid
        clean = raw.loc[~bad].set_index("time").sort_index()
        hour = clean[["power", "wind", "temp"]].resample("h").mean()
        hour["n_samples"] = clean.power.resample("h").count()
        all_hours = pd.date_range(
            raw.time.min().floor("h"), raw.time.max().floor("h"), freq="h"
        )
        hour = hour.reindex(all_hours)
        hour["n_samples"] = hour.n_samples.fillna(0).astype(int)
        # No interpolation of target. Every accepted hour has six ten-minute samples.
        hour.loc[hour.n_samples != 6, ["power", "wind", "temp"]] = np.nan
        hour["turbine_id"] = tid
        hour.index.name = "valid_local"
        frames.append(hour.reset_index())
        audit[tid] = dict(
            file=path.name,
            sha256=digest(path),
            rows=len(raw),
            start=str(raw.time.min()),
            end=str(raw.time.max()),
            invalid_rows=int(bad.sum()),
            total_hours=len(hour),
            complete_hours=int((hour.n_samples == 6).sum()),
            incomplete_hours=int((hour.n_samples != 6).sum()),
            february_rows=int(
                ((raw.time >= "2026-02-01") & (raw.time < "2026-03-01")).sum()
            ),
            off_grid_rows=int(off_grid.sum()),
            missing_10min_slots=int(
                len(
                    pd.date_range(
                        raw.time.min().floor("10min"),
                        raw.time.max().floor("10min"),
                        freq="10min",
                    ).difference(raw.time)
                )
            ),
        )
    data = pd.concat(frames, ignore_index=True)
    (ROOT / "data").mkdir(exist_ok=True)
    data.to_csv(ROOT / "data/hourly.csv", index=False)
    dump(ROOT / "reports/data_audit.json", audit)
    return data


def read_weather():
    return load_weather(tolerant=False)


def load_weather(tolerant=False):
    df, manifest, problems = read_archive(ROOT, COORDS, tolerant=tolerant)
    dump(ROOT / "reports/weather_manifest.json", manifest)
    dump(ROOT / "reports/weather_cache_problems.json", problems)
    for problem in problems:
        warnings.warn(problem, RuntimeWarning)
    df["valid_local"] = df.valid_utc.dt.tz_localize(None) + OFFSET
    return df


def features(frame):
    x = frame.copy()
    x["direction_sin"] = np.sin(np.deg2rad(x.direction))
    x["direction_cos"] = np.cos(np.deg2rad(x.direction))
    for name, values, period in [
        ("hour", x.valid_local.dt.hour, 24),
        ("month", x.valid_local.dt.month, 12),
    ]:
        x[name + "_sin"] = np.sin(2 * np.pi * values / period)
        x[name + "_cos"] = np.cos(2 * np.pi * values / period)
    return x[FEATURES].astype(float)


def safe_offset(lead_hours):
    """Use weather with at least 12 hours between nominal init and issue."""
    return np.maximum(
        1, np.ceil((np.asarray(lead_hours) + LATENCY_HOURS) / 24).astype(int)
    )


def schedule(first_issue, last_issue):
    first, last = pd.Timestamp(first_issue), pd.Timestamp(last_issue)
    if pd.isna(first) or pd.isna(last) or first.tzinfo or last.tzinfo:
        raise ValueError("Issue dates must be local dates in YYYY-MM-DD format")
    if first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError("Issue dates must be midnight dates, first not after last")
    rows = []
    for day in pd.date_range(first_issue, last_issue, freq="D"):
        issue_local = day.normalize() + pd.Timedelta(hours=23)
        issue_utc = (issue_local - OFFSET).tz_localize("UTC")
        for lead in range(1, 49):
            valid = issue_utc + pd.Timedelta(hours=lead)
            days = int(safe_offset(lead))
            for tid in COORDS:
                rows.append(
                    dict(
                        issue_utc=issue_utc,
                        valid_utc=valid,
                        valid_local=valid.tz_localize(None) + OFFSET,
                        lead_hours=lead,
                        turbine_id=tid,
                        offset_days=days,
                        availability_bound_utc=valid
                        - pd.Timedelta(days=days)
                        + pd.Timedelta(hours=LATENCY_HOURS),
                    )
                )
    frame = pd.DataFrame(rows)
    if not (frame.availability_bound_utc <= frame.issue_utc).all():
        raise ValueError("Future weather detected")
    return frame


def forecast_frame(plan, weather):
    return plan.merge(
        weather.drop(columns="valid_local"),
        on=["valid_utc", "turbine_id", "offset_days"],
        how="left",
        validate="many_to_one",
    )


def metrics(y, pred):
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = np.asarray(y)[valid], np.asarray(pred)[valid]
    if len(y) == 0:
        return dict(n=0, MAE=None, RMSE=None, bias=None)
    return dict(
        n=len(y),
        MAE=float(np.mean(np.abs(y - pred))),
        RMSE=float(np.sqrt(np.mean((y - pred) ** 2))),
        bias=float(np.mean(pred - y)),
    )


def predict_with_fallback(frame, model, means):
    good = valid_weather(frame)
    prediction = frame.turbine_id.map(means).astype(float)
    if prediction.isna().any():
        raise ValueError("Missing historical mean for a turbine; retrain the model")
    with threadpool_limits(limits=2):
        if good.any():
            prediction.loc[good] = np.clip(
                model.predict(features(frame.loc[good])), 0, 1
            )
    return prediction, good


def train():
    scada = pd.read_csv(ROOT / "data/hourly.csv", parse_dates=["valid_local"])
    if scada.duplicated(["valid_local", "turbine_id"]).any():
        raise ValueError("Duplicate hourly targets; run prepare again")
    if not scada.loc[scada.power.notna(), "n_samples"].eq(6).all():
        raise ValueError("Training targets must contain six measurements per hour")
    weather = read_weather()
    weather = weather[valid_weather(weather) & (weather.valid_local >= "2024-04-01")]
    pairs = weather.merge(
        scada[["valid_local", "turbine_id", "power"]],
        on=["valid_local", "turbine_id"],
        validate="many_to_one",
    ).dropna(subset=["power", "wind", "direction", "temp"])
    selection_ready = pd.Timestamp("2025-11-30 23:00")
    train_rows = pairs[completed_before(pairs, selection_ready)]
    selection_means = (
        scada[completed_before(scada, selection_ready)]
        .groupby("turbine_id")
        .power.mean()
        .to_dict()
    )
    dec = forecast_frame(schedule("2025-11-30", "2025-12-30"), weather)
    dec = dec.merge(
        scada[["valid_local", "turbine_id", "power"]],
        on=["valid_local", "turbine_id"],
        how="left",
    )
    dec = dec[completed_before(dec, POINT_READY_LOCAL)].dropna(subset=["power"])
    if len(train_rows) < 5000 or len(dec) < 500:
        raise ValueError("Insufficient training/validation data")
    candidates = [(15, 180, 0.06), (31, 180, 0.06), (15, 300, 0.04)]
    scores = []
    with threadpool_limits(limits=2):
        for leaves, iterations, lr in candidates:
            model = HistGradientBoostingRegressor(
                max_leaf_nodes=leaves,
                max_iter=iterations,
                learning_rate=lr,
                min_samples_leaf=40,
                l2_regularization=5,
                early_stopping=False,
                random_state=42,
            )
            model.fit(features(train_rows), train_rows.power)
            prediction, _ = predict_with_fallback(dec, model, selection_means)
            score = metrics(dec.power, prediction)
            scores.append(dict(leaves=leaves, iterations=iterations, lr=lr, **score))
            print("December validation", scores[-1], flush=True)
        best = min(scores, key=lambda v: v["MAE"])
        final_rows = pairs[completed_before(pairs, POINT_READY_LOCAL)]
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=best["leaves"],
            max_iter=best["iterations"],
            learning_rate=best["lr"],
            min_samples_leaf=40,
            l2_regularization=5,
            early_stopping=False,
            random_state=42,
        ).fit(features(final_rows), final_rows.power)
    jan = forecast_frame(schedule("2025-12-31", "2026-01-30"), weather)
    jan = jan[jan.valid_local < "2026-02-01"].merge(
        scada[["valid_local", "turbine_id", "power"]],
        on=["valid_local", "turbine_id"],
        how="left",
    )
    means = (
        scada[completed_before(scada, POINT_READY_LOCAL)]
        .groupby("turbine_id")
        .power.mean()
        .to_dict()
    )
    if set(means) != set(COORDS) or not np.isfinite(list(means.values())).all():
        raise ValueError(
            "Both turbines require historical targets before the first issue"
        )
    jan["prediction"], good = predict_with_fallback(jan, model, means)
    jan["status"] = np.where(good, "weather_model", "fallback_climatology")
    jan["climatology"] = jan.turbine_id.map(means)
    # Persistence uses only full hours completed before each historical issue.
    jan["persistence"] = np.nan
    for (issue, tid), part in jan.groupby(["issue_utc", "turbine_id"]):
        issue_local = issue.tz_localize(None) + OFFSET
        hist = scada[
            (scada.turbine_id == tid)
            & (scada.valid_local + pd.Timedelta(hours=1) <= issue_local)
        ].dropna(subset=["power"])
        jan.loc[part.index, "persistence"] = (
            hist.power.iloc[-1] if len(hist) else means[tid]
        )
    results = []
    for tid in COORDS:
        for horizon in ["1-24", "25-48"]:
            part = jan[
                (jan.turbine_id == tid)
                & (
                    (jan.lead_hours <= 24)
                    if horizon == "1-24"
                    else (jan.lead_hours > 24)
                )
            ]
            for method in ["prediction", "climatology", "persistence"]:
                results.append(
                    dict(
                        turbine_id=tid,
                        horizon=horizon,
                        model=method,
                        scheduled_rows=len(part),
                        missing_targets=int(part.power.isna().sum()),
                        fallback_rows=int(
                            (part.status == "fallback_climatology").sum()
                        ),
                        **metrics(part.power, part[method]),
                    )
                )
    radii = {}
    for tid in COORDS:
        for h in [1, 2]:
            part = jan[
                (jan.turbine_id == tid)
                & (((jan.lead_hours - 1) // 24 + 1) == h)
                & completed_before(jan, CALIBRATION_READY_LOCAL)
                & (jan.status == "weather_model")
            ].dropna(subset=["power"])
            if len(part) < 100:
                raise ValueError(
                    f"Insufficient January calibration for turbine {tid}, horizon {h}"
                )
            errors = np.abs(part.power - part.prediction)
            # Empirical temporal calibration; no exchangeability/coverage guarantee claimed.
            radii[f"{tid}_{h}"] = float(np.quantile(errors, 0.9, method="higher"))
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models/forecast.joblib"
    temporary = model_path.with_suffix(".tmp")
    joblib.dump(
        dict(
            model=model,
            means=means,
            radii=radii,
            features=FEATURES,
            version=MODEL_VERSION,
            settings_signature=settings_signature(),
            ready_local=str(CALIBRATION_READY_LOCAL),
            point_ready_local=str(POINT_READY_LOCAL),
        ),
        temporary,
    )
    temporary.replace(model_path)
    jan.to_csv(ROOT / "reports/january_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(ROOT / "reports/january_metrics.csv", index=False)
    dump(
        ROOT / "reports/model_card.json",
        dict(
            selection=scores,
            selected=best,
            training_rows=len(final_rows),
            training_unique_hours=final_rows[["valid_local", "turbine_id"]]
            .drop_duplicates()
            .shape[0],
            train_target_end=str(final_rows.valid_local.max()),
            target_available_at=str(
                final_rows.valid_local.max() + pd.Timedelta(hours=1)
            ),
            point_model_ready_local=str(POINT_READY_LOCAL),
            replay_ready_local=str(CALIBRATION_READY_LOCAL),
            validation="2025-12 (completed hours only)",
            test_and_interval_calibration="2026-01",
            interval_radii=radii,
            features=FEATURES,
            target="mean normalized active power",
            timezone_offset_hours=CONFIG["utc_offset_hours"],
            time_assumptions_confirmed=CONFIG["time_assumptions_confirmed"],
            weather_latency_assumption_hours=LATENCY_HOURS,
            model_sha256=digest(model_path),
            hourly_sha256=digest(ROOT / "data/hourly.csv"),
            code_sha256={
                p: digest(ROOT / p)
                for p in ["wind_agent.py", "weather_archive.py", "config.json"]
            },
            january_fallback_rows=int((~good).sum()),
            archive_provenance="Fixed lead offsets; publication timestamps not supplied by API",
        ),
    )
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def results_directory(first_issue, last_issue):
    first = str(pd.Timestamp(first_issue).date())
    last = str(pd.Timestamp(last_issue).date())
    if (first, last) == FULL_REPLAY:
        return ROOT / "results"
    if first == last:
        return ROOT / "results/cycles" / first
    return ROOT / "results/runs" / f"{first}_{last}"


def load_model():
    path = ROOT / "models/forecast.joblib"
    if not path.exists():
        raise ValueError("Model is missing. Run: python wind_agent.py train")
    bundle = joblib.load(path)
    if bundle.get("version") != MODEL_VERSION:
        raise ValueError("Old model format. Run train again before replay")
    if bundle.get("settings_signature") != settings_signature():
        raise ValueError(
            "Configuration differs from the trained model. Run train again"
        )
    if bundle.get("features") != FEATURES:
        raise ValueError("Model features do not match the code. Run train again")
    return bundle


def append_events(events):
    path = ROOT / "results/agent_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def replay(
    first_issue="2026-01-31",
    last_issue="2026-02-28",
    refresh=False,
    offline=False,
    skip_unchanged=False,
    require_verified_availability=False,
):
    """Fetch -> validate -> predict -> review -> publish, with an explicit fallback."""
    plan = schedule(first_issue, last_issue)
    if refresh and offline:
        raise ValueError("--refresh and --offline cannot be used together")
    if require_verified_availability:
        raise ValueError(
            "Previous Runs does not provide historical publication timestamps. "
            "Verified availability requires another weather archive adapter."
        )
    bundle = load_model()
    first_local = plan.issue_utc.min().tz_localize(None) + OFFSET
    if first_local < pd.Timestamp(bundle["ready_local"]):
        raise ValueError(
            "Model/calibration was not available at the requested historical issue"
        )

    events = []

    def event(state, **details):
        events.append(
            dict(
                recorded_at=datetime.now(timezone.utc).isoformat(),
                state=state,
                **details,
            )
        )

    event("PLAN", first_issue=first_issue, last_issue=last_issue)
    event(
        "FETCH",
        mode="offline" if offline else "refresh" if refresh else "cache_or_download",
    )
    if not offline:
        # A failure in one month/turbine must not prevent fetching the other one.
        for first, last in month_ranges(
            plan.valid_utc.min().date(), plan.valid_utc.max().date()
        ):
            for tid in COORDS:
                try:
                    fetch_month(tid, first, last, refresh)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    event(
                        "FETCH_FAILED_USE_CACHE",
                        turbine_id=tid,
                        month=str(first),
                        error=str(exc),
                    )

    weather = load_weather(tolerant=True)
    output = forecast_frame(plan, weather)
    model_hash = digest(ROOT / "models/forecast.joblib")
    source_hash = hashlib.sha256(
        (digest(ROOT / "wind_agent.py") + digest(ROOT / "weather_archive.py")).encode()
    ).hexdigest()
    input_columns = [
        "issue_utc",
        "valid_utc",
        "turbine_id",
        "wind",
        "direction",
        "temp",
        "offset_days",
    ]
    fingerprint = hashlib.sha256(
        (
            model_hash
            + source_hash
            + settings_signature()
            + output[input_columns].to_csv(index=False)
        ).encode()
    ).hexdigest()
    results_dir = results_directory(first_issue, last_issue)
    summary_path = results_dir / "run_summary.json"
    if skip_unchanged and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_files = [
            "forecasts_48h.csv",
            "submission_february.csv",
            "weather_manifest.json",
        ]
        if previous.get("february_rows", 0):
            expected_files.append("station_february.csv")
        if previous.get("input_fingerprint") == fingerprint and all(
            (results_dir / name).exists() for name in expected_files
        ):
            event(
                "UNCHANGED",
                input_fingerprint=fingerprint,
                action="keep_existing_forecast",
            )
            append_events(events)
            print("Input unchanged; existing forecast kept.", flush=True)
            return previous

    output["prediction"], good = predict_with_fallback(
        output, bundle["model"], bundle["means"]
    )
    output["status"] = np.where(good, "weather_model", "fallback_climatology")
    output["availability_status"] = "assumed_delay_not_verified_publication"
    output["uncertainty_radius"] = [
        bundle["radii"][f"{tid}_{(lead - 1) // 24 + 1}"]
        for tid, lead in zip(output.turbine_id, output.lead_hours)
    ]
    output["lower90"] = (output.prediction - output.uncertainty_radius).clip(0, 1)
    output["upper90"] = (output.prediction + output.uncertainty_radius).clip(0, 1)
    output.loc[~good, ["lower90", "upper90"]] = [0, 1]
    output["ramp_flag"] = False
    for issue, part in output.groupby("issue_utc", sort=True):
        for _, turbine in part.groupby("turbine_id"):
            output.loc[turbine.index, "ramp_flag"] = (
                turbine.prediction.diff().abs().gt(0.3)
            )
        event(
            "VALIDATE",
            issue_utc=str(issue),
            hours_per_turbine=48,
            missing_weather=int((part.status != "weather_model").sum()),
        )
        event("PREDICT", issue_utc=str(issue), model_sha256=model_hash)
        event(
            "REVIEW",
            issue_utc=str(issue),
            action=(
                "publish"
                if (part.status == "weather_model").all()
                else "publish_degraded"
            ),
            min_prediction=float(part.prediction.min()),
            max_prediction=float(part.prediction.max()),
        )

    if (
        len(output) != len(plan)
        or output.duplicated(["issue_utc", "valid_utc", "turbine_id"]).any()
    ):
        raise ValueError("Forecast coverage or uniqueness check failed")
    if (
        not np.isfinite(output.prediction).all()
        or not output.prediction.between(0, 1).all()
    ):
        raise ValueError("Prediction is not finite or is outside the normalized range")
    submission = output[
        (output.lead_hours <= 24)
        & (output.valid_local >= "2026-02-01")
        & (output.valid_local < "2026-03-01")
    ].copy()
    if (
        str(pd.Timestamp(first_issue).date()),
        str(pd.Timestamp(last_issue).date()),
    ) == FULL_REPLAY:
        if (
            len(submission) != 1344
            or submission.duplicated(["valid_local", "turbine_id"]).any()
        ):
            raise ValueError(
                "February submission must have 672 unique hours per turbine"
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_dir / "forecasts_48h.csv", index=False)
    submission.to_csv(results_dir / "submission_february.csv", index=False)
    if not submission.empty:
        station = submission.pivot(
            index="valid_local", columns="turbine_id", values="prediction"
        )
        station = station.rename(
            columns={1: "turbine_1_normalized", 2: "turbine_2_normalized"}
        )
        station["equal_capacity_mean_assumption"] = station.mean(axis=1, skipna=False)
        station.to_csv(results_dir / "station_february.csv")
    manifest = json.loads(
        (ROOT / "reports/weather_manifest.json").read_text(encoding="utf-8")
    )
    dump(results_dir / "weather_manifest.json", manifest)
    summary = dict(
        rows=len(output),
        february_rows=len(submission),
        fallback_rows=int((~good).sum()),
        issues=int(output.issue_utc.nunique()),
        model_sha256=model_hash,
        code_sha256=source_hash,
        input_fingerprint=fingerprint,
        settings_signature=settings_signature(),
        first_issue=first_issue,
        last_issue=last_issue,
        weather_cache_problems=json.loads(
            (ROOT / "reports/weather_cache_problems.json").read_text()
        ),
        historical_availability_verified=False,
        time_assumptions_confirmed=CONFIG["time_assumptions_confirmed"],
        provenance_caveat="availability_bound_utc is assumed, not observed publication time",
    )
    dump(summary_path, summary)
    event(
        "PUBLISH",
        rows=len(output),
        february_day_ahead_rows=len(submission),
        fallback_rows=int((~good).sum()),
        input_fingerprint=fingerprint,
    )
    append_events(events)
    print(
        f"Published {len(output)} forecasts; fallback rows: {int((~good).sum())}",
        flush=True,
    )
    return summary


def package_results():
    """Package only the completed full replay and bind it to current reports."""
    summary_path = ROOT / "results/run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("first_issue"), summary.get("last_issue")) != FULL_REPLAY:
        raise ValueError("Run the full February replay before packaging")
    if summary["rows"] != 2784 or summary["february_rows"] != 1344:
        raise ValueError("Full replay coverage is incomplete")
    if digest(ROOT / "models/forecast.joblib") != summary["model_sha256"]:
        raise ValueError("Model changed since replay; repeat replay before packaging")
    current_code = hashlib.sha256(
        (digest(ROOT / "wind_agent.py") + digest(ROOT / "weather_archive.py")).encode()
    ).hexdigest()
    if (
        current_code != summary["code_sha256"]
        or settings_signature() != summary["settings_signature"]
    ):
        raise ValueError(
            "Code/configuration changed since replay; repeat replay before packaging"
        )
    files = [
        "forecasts_48h.csv",
        "submission_february.csv",
        "station_february.csv",
        "weather_manifest.json",
        "run_summary.json",
    ]
    path = ROOT / "results/forecast_results.zip"
    with zipfile.ZipFile(
        path.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in files:
            archive.write(ROOT / "results" / name, arcname="results/" + name)
    path.with_suffix(".tmp").replace(path)
    tracked = [
        "README.md",
        "requirements.txt",
        "config.json",
        "wind_agent.py",
        "weather_archive.py",
        "tests/test_agent.py",
        "tests/test_regressions.py",
        "reports/data_audit.json",
        "reports/model_card.json",
        "reports/january_metrics.csv",
        "reports/weather_manifest.json",
        "results/run_summary.json",
        "results/forecast_results.zip",
    ]
    dump(
        ROOT / "reports/artifact_checksums.json",
        {name: digest(ROOT / name) for name in tracked if (ROOT / name).exists()},
    )
    print("Packaged:", path, flush=True)


def status():
    """Explain which step is available, without downloading weather or changing files."""
    for label, path in [
        ("Hourly data", ROOT / "data/hourly.csv"),
        ("Trained model", ROOT / "models/forecast.joblib"),
        ("Full replay", ROOT / "results/run_summary.json"),
    ]:
        print(f'{label}: {"present" if path.exists() else "missing"}')
    print("Weather cache files:", len(list((ROOT / "cache").glob("gfs_t*.json"))))
    print("Time assumption confirmed:", CONFIG["time_assumptions_confirmed"])
    print("Historical weather publication time: not verified by this provider")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "prepare", help="Проверить CSV и получить полные почасовые измерения"
    )
    command.add_argument(
        "--data-dir", default="raw", help="Папка с двумя исходными CSV"
    )
    command = sub.add_parser(
        "fetch", help="Скачать и проверить месячный архив прогнозов погоды"
    )
    command.add_argument("--start", default="2024-04-01")
    command.add_argument("--end", default="2026-03-02")
    command.add_argument("--refresh", action="store_true")
    sub.add_parser("train", help="Обучить модель, проверить январь, сохранить метрики")
    command = sub.add_parser(
        "replay", help="Повторить ежедневные прогнозы за выбранные даты"
    )
    command.add_argument("--first-issue", default=FULL_REPLAY[0])
    command.add_argument("--last-issue", default=FULL_REPLAY[1])
    network = command.add_mutually_exclusive_group()
    network.add_argument("--refresh", action="store_true")
    network.add_argument("--offline", action="store_true")
    command.add_argument(
        "--require-verified-availability",
        action="store_true",
        help="Отказаться от расчёта без подтверждённого времени публикации погоды",
    )
    command = sub.add_parser(
        "watch", help="Проверять обновления выбранного исторического выпуска"
    )
    command.add_argument("--issue-date", required=True)
    command.add_argument("--interval-seconds", type=int, default=3600)
    sub.add_parser("status", help="Показать наличие данных, модели и результатов")
    sub.add_parser(
        "package", help="Упаковать полный пересчёт и обновить контрольные суммы"
    )
    command = sub.add_parser(
        "run", help="Полный цикл: подготовка, погода, обучение, февраль, архив"
    )
    command.add_argument("--data-dir", default="raw")
    command.add_argument(
        "--offline",
        action="store_true",
        help="Использовать уже загруженный погодный архив",
    )
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.data_dir)
            print("Prepared: data/hourly.csv; audit: reports/data_audit.json")
        elif args.command == "fetch":
            fetch_range(args.start, args.end, args.refresh)
            read_weather()
        elif args.command == "train":
            train()
        elif args.command == "replay":
            replay(
                args.first_issue,
                args.last_issue,
                args.refresh,
                args.offline,
                require_verified_availability=args.require_verified_availability,
            )
        elif args.command == "status":
            status()
        elif args.command == "package":
            package_results()
        elif args.command == "run":
            prepare(args.data_dir)
            if not args.offline:
                fetch_range("2024-04-01", "2026-03-02")
            train()
            replay(offline=args.offline)
            package_results()
        else:
            if args.interval_seconds < 60:
                parser.error("Minimum interval is 60 seconds")
            while True:
                replay(
                    args.issue_date, args.issue_date, refresh=True, skip_unchanged=True
                )
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
