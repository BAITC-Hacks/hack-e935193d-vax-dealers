"""Загрузка и проверка архивных прогнозов Open-Meteo Previous Runs.

Ряды previous_dayN не содержат фактического времени публикации выпуска.
Этот модуль не выдаёт время скачивания за историческую доступность.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ("wind_speed_80m", "wind_direction_80m", "temperature_2m")
UNITS = ("m/s", "°", "°C")
WEATHER_COLUMNS = [
    "valid_utc",
    "turbine_id",
    "offset_days",
    "wind",
    "direction",
    "temp",
]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_payload(data, start, end):
    """Reject wrong units, missing columns, duplicate hours and truncated responses."""
    if data.get("utc_offset_seconds") != 0:
        raise ValueError("Weather must use UTC (utc_offset_seconds=0)")
    hourly = data.get("hourly", {})
    units = data.get("hourly_units", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="raise")
    expected = pd.date_range(
        str(start),
        pd.Timestamp(end) + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
        tz="UTC",
    )
    if not times.equals(expected):
        raise ValueError(
            "Weather timestamps are incomplete, duplicated or out of order"
        )
    for days in (1, 2, 3):
        for variable, unit in zip(VARIABLES, UNITS):
            name = f"{variable}_previous_day{days}"
            if units.get(name) != unit:
                raise ValueError(f"Wrong weather units: {name}, expected {unit}")
            if len(hourly.get(name, [])) != len(times):
                raise ValueError(f"Missing or truncated weather column: {name}")


def fetch_month(root, coordinates, tid, start, end, refresh=False):
    path = Path(root) / "cache" / f"gfs_t{tid}_{start}_{end}.json"
    lat, lon = coordinates[tid]
    if path.exists() and not refresh:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            validate_payload(cached, start, end)
            provenance = cached.get("_provenance", {})
            if provenance.get("lat") == lat and provenance.get("lon") == lon:
                return path
        except (ValueError, KeyError, TypeError, OSError):
            pass  # Replace only after a valid replacement has been downloaded.
    query = dict(
        latitude=lat,
        longitude=lon,
        start_date=str(start),
        end_date=str(end),
        hourly=",".join(f"{v}_previous_day{d}" for d in (1, 2, 3) for v in VARIABLES),
        models="gfs_seamless",
        timezone="GMT",
        wind_speed_unit="ms",
    )
    url = (
        "https://previous-runs-api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode(query)
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                raw = response.read()
            data = json.loads(raw)
            validate_payload(data, start, end)
            data["_provenance"] = dict(
                url=url,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                sha256_payload=hashlib.sha256(raw).hexdigest(),
                lat=lat,
                lon=lon,
                provider="Open-Meteo Previous Runs / NOAA GFS",
                publication_time_verified=False,
            )
            atomic_json(path, data)
            return path
        except (OSError, ValueError, KeyError, TypeError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def month_ranges(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Weather start date must not be after end date")
    # One cache file per calendar month, even for a one-day replay.
    day = start.normalize().replace(day=1)
    while day <= end:
        last = day + pd.offsets.MonthEnd(0)
        yield day.date(), last.date()
        day = last + pd.Timedelta(days=1)


def empty_weather():
    frame = pd.DataFrame(columns=WEATHER_COLUMNS)
    frame["valid_utc"] = pd.to_datetime(frame.valid_utc, utc=True)
    for name in WEATHER_COLUMNS[1:]:
        frame[name] = frame[name].astype(float)
    return frame


def read_archive(root, coordinates, tolerant=False):
    records, manifest, problems = [], [], []
    snapshots = []
    for path in sorted((Path(root) / "cache").glob("gfs_t*.json")):
        try:
            _, turbine, start, end = path.stem.split("_")
            tid = int(turbine[1:])
            data = json.loads(path.read_text(encoding="utf-8"))
            validate_payload(data, start, end)
            provenance = data.get("_provenance", {})
            lat, lon = coordinates[tid]
            if provenance.get("lat") != lat or provenance.get("lon") != lon:
                raise ValueError("Cached weather coordinates do not match this turbine")
            retrieved = pd.Timestamp(provenance["retrieved_at"])
            if pd.isna(retrieved) or retrieved.tzinfo is None:
                raise ValueError("Cache retrieval time must include a timezone")
            snapshots.append((retrieved, path, tid, data))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            message = f"{path.name}: {exc}"
            if not tolerant:
                raise ValueError(message) from exc
            problems.append(message)
    for _, path, tid, data in sorted(
        snapshots, key=lambda item: (item[0], item[1].name)
    ):
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"], utc=True)
        for days in (1, 2, 3):
            frame = pd.DataFrame(
                dict(
                    valid_utc=times,
                    turbine_id=tid,
                    offset_days=days,
                    wind=hourly[f"wind_speed_80m_previous_day{days}"],
                    direction=hourly[f"wind_direction_80m_previous_day{days}"],
                    temp=hourly[f"temperature_2m_previous_day{days}"],
                )
            )
            for column in ("wind", "direction", "temp"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            records.append(frame)
        manifest.append(
            dict(
                file=path.name,
                sha256_file=hashlib.sha256(path.read_bytes()).hexdigest(),
                grid_latitude=data.get("latitude"),
                grid_longitude=data.get("longitude"),
                **data["_provenance"],
            )
        )
    if not records:
        if not tolerant:
            raise ValueError(
                "No valid weather archive. Run: python wind_agent.py fetch"
            )
        return empty_weather(), manifest, problems
    frame = pd.concat(records, ignore_index=True)
    frame = frame.drop_duplicates(
        ["valid_utc", "turbine_id", "offset_days"], keep="last"
    )
    return frame, manifest, problems


def valid_weather(frame):
    """Use the same finite-value/range policy during training and replay."""
    return (
        np.isfinite(frame[["wind", "direction", "temp"]]).all(axis=1)
        & frame.wind.between(0, 80)
        & frame.direction.between(0, 360)
        & frame.temp.between(-70, 70)
    )
