"""Проверка сохранённого прогноза без загрузки модели и без сетевых запросов."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

RESULT_FILES = (
    "forecasts_48h.csv",
    "submission_february.csv",
    "station_february.csv",
    "weather_manifest.json",
    "run_summary.json",
)
HASHED_FILES = RESULT_FILES[:-1]


def file_hashes(directory):
    return {
        name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        for name in HASHED_FILES
        if (Path(directory) / name).exists()
    }


def load_snapshot(root, archive_only=False):
    """Choose one complete source; never mix archived CSVs with newer metadata."""
    directory = Path(root) / "results"
    if not archive_only and all((directory / name).is_file() for name in RESULT_FILES):
        return {name: (directory / name).read_bytes() for name in RESULT_FILES}, "files"
    archive = directory / "forecast_results.zip"
    if not archive.exists():
        raise ValueError(
            "Нет полного расчёта. Выполните replay или скачайте forecast_results.zip"
        )
    try:
        with zipfile.ZipFile(archive) as stream:
            # Read only known members; extraction/path traversal is unnecessary.
            content = {name: stream.read("results/" + name) for name in RESULT_FILES}
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Архив результатов повреждён или неполон") from exc
    return content, "archive"


def inspect_snapshot(content):
    errors, warnings = [], []
    counts = {}
    try:
        summary = json.loads(content["run_summary.json"])
        frame = pd.read_csv(io.BytesIO(content["forecasts_48h.csv"]))
        submission = pd.read_csv(io.BytesIO(content["submission_february.csv"]))
        station = pd.read_csv(io.BytesIO(content["station_february.csv"]))
        manifest = json.loads(content["weather_manifest.json"])
        if not isinstance(manifest, list):
            raise ValueError("Манифест погоды должен содержать список источников")
        required = {
            "issue_utc",
            "valid_utc",
            "valid_local",
            "turbine_id",
            "lead_hours",
            "prediction",
            "lower90",
            "upper90",
            "status",
            "availability_bound_utc",
        }
        for name, table in [("48 часов", frame), ("февраль", submission)]:
            if not required.issubset(table.columns):
                raise ValueError(f"В таблице {name} отсутствуют обязательные столбцы")
            for column in ("issue_utc", "valid_utc", "availability_bound_utc"):
                table[column] = pd.to_datetime(table[column], utc=True, errors="raise")
            table["valid_local"] = pd.to_datetime(table.valid_local, errors="raise")
            if table[list(required)].isna().any().any():
                errors.append(f"В обязательных полях таблицы {name} есть пропуски")
            if table.duplicated(["issue_utc", "valid_utc", "turbine_id"]).any():
                errors.append(f"Повторяются ключи прогнозов: {name}")
            values = table[["prediction", "lower90", "upper90"]].to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or not ((values >= 0) & (values <= 1)).all()
            ):
                errors.append(f"Некорректный диапазон мощности: {name}")
            if not (
                (table.lower90 <= table.prediction)
                & (table.prediction <= table.upper90)
            ).all():
                errors.append(f"Диапазон неопределённости не включает прогноз: {name}")
            if not table.status.isin(["weather_model", "fallback_climatology"]).all():
                errors.append(f"Неизвестный режим прогноза: {name}")
            fallback = table.status.eq("fallback_climatology")
            if not (
                table.loc[fallback, "lower90"].eq(0)
                & table.loc[fallback, "upper90"].eq(1)
            ).all():
                errors.append(f"Резервный прогноз должен иметь диапазон [0,1]: {name}")
        if set(frame.turbine_id) != {1, 2}:
            errors.append("Ожидаются ровно две турбины: 1 и 2")
        offset = pd.Timedelta(hours=summary["utc_offset_hours"])
        issue_days = pd.date_range(
            summary["first_issue"], summary["last_issue"], freq="D"
        )
        expected_issues = (issue_days + pd.Timedelta(hours=23) - offset).tz_localize(
            "UTC"
        )
        if set(frame.issue_utc) != set(expected_issues):
            errors.append("Набор дат выпуска не соответствует периоду расчёта")
        for (_, tid), part in frame.groupby(["issue_utc", "turbine_id"]):
            if len(part) != 48 or set(part.lead_hours) != set(range(1, 49)):
                errors.append(f"Неполный горизонт 1–48 часов для турбины {tid}")
                break
        if len(frame) != len(expected_issues) * 96:
            errors.append("Число строк не соответствует двум турбинам и 48 часам")
        lead = (frame.valid_utc - frame.issue_utc).dt.total_seconds() / 3600
        if not np.array_equal(lead.to_numpy(), frame.lead_hours.to_numpy()):
            errors.append("Горизонт не совпадает с разностью времён")
        if not (
            (frame.valid_utc.dt.tz_localize(None) + offset) == frame.valid_local
        ).all():
            errors.append("Местное время не соответствует сохранённому UTC-сдвигу")
        if not (frame.availability_bound_utc <= frame.issue_utc).all():
            errors.append("Нарушена расчётная граница доступности погоды")

        expected = frame[
            (frame.lead_hours <= 24)
            & (frame.valid_local >= "2026-02-01")
            & (frame.valid_local < "2026-03-01")
        ]
        keys = ["issue_utc", "valid_utc", "turbine_id"]
        columns = keys + [
            "valid_local",
            "lead_hours",
            "prediction",
            "lower90",
            "upper90",
            "status",
        ]
        try:
            pd.testing.assert_frame_equal(
                expected[columns].sort_values(keys).reset_index(drop=True),
                submission[columns].sort_values(keys).reset_index(drop=True),
                check_dtype=False,
                atol=1e-12,
                rtol=0,
            )
        except AssertionError:
            errors.append(
                "Февральская подача не совпадает с первыми 24 часами полных прогнозов"
            )
        if (summary["first_issue"], summary["last_issue"]) != (
            "2026-01-31",
            "2026-02-28",
        ):
            errors.append("Для сдачи нужен полный расчёт 31 января — 28 февраля")
        if (
            len(submission) != 1344
            or submission.duplicated(["valid_local", "turbine_id"]).any()
        ):
            errors.append(
                "В феврале должно быть ровно 672 уникальных часа каждой турбины"
            )
        february = set(
            pd.date_range("2026-02-01", "2026-03-01", freq="h", inclusive="left")
        )
        if any(
            set(submission[submission.turbine_id == tid].valid_local) != february
            for tid in (1, 2)
        ):
            errors.append("Февральская сетка времени содержит пропуски или лишние часы")
        station["valid_local"] = pd.to_datetime(station.valid_local)
        pivot = submission.pivot(
            index="valid_local", columns="turbine_id", values="prediction"
        )
        check_station = station.set_index("valid_local").sort_index()
        if not check_station.index.equals(pivot.index):
            errors.append("Время в таблице станции не совпадает с февральской подачей")
        else:
            for tid in (1, 2):
                if not np.allclose(
                    check_station[f"turbine_{tid}_normalized"],
                    pivot[tid],
                    atol=1e-12,
                    rtol=0,
                ):
                    errors.append(
                        f"Таблица станции не совпадает с прогнозом турбины {tid}"
                    )
            if not np.allclose(
                check_station.equal_capacity_mean_assumption,
                pivot.mean(axis=1),
                atol=1e-12,
                rtol=0,
            ):
                errors.append("Неверно рассчитано условное среднее станции")
        counts = dict(
            forecast_rows=len(frame),
            february_rows=len(submission),
            issues=int(frame.issue_utc.nunique()),
            fallback_rows=int(frame.status.eq("fallback_climatology").sum()),
        )
        for key, count in [
            ("rows", len(frame)),
            ("february_rows", len(submission)),
            ("issues", counts["issues"]),
            ("fallback_rows", counts["fallback_rows"]),
        ]:
            if summary.get(key) != count:
                errors.append(f"Итоговый отчёт содержит неверное значение {key}")
        hashes = summary.get("artifact_sha256", {})
        for name in HASHED_FILES:
            if hashlib.sha256(content[name]).hexdigest() != hashes.get(name):
                errors.append(f"Контрольная сумма отсутствует или не совпадает: {name}")
        if not summary.get("time_assumptions_confirmed"):
            warnings.append(
                "Часовой пояс и смысл временных меток CSV не подтверждены организаторами"
            )
        if not summary.get("historical_availability_verified"):
            warnings.append(
                "Историческое время публикации погодных выпусков не подтверждено"
            )
        if counts["fallback_rows"]:
            warnings.append(f"Резервных прогнозов: {counts['fallback_rows']}")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"Не удалось проверить структуру результатов: {exc}")
    return dict(
        valid=not errors,
        assumptions_confirmed=not errors and not warnings,
        errors=list(dict.fromkeys(errors)),
        warnings=warnings,
        counts=counts,
    )


def check_results(root, archive_only=False):
    try:
        content, source = load_snapshot(root, archive_only)
        report = inspect_snapshot(content)
        report["source"] = source
        return report
    except (ValueError, OSError) as exc:
        return dict(
            valid=False,
            assumptions_confirmed=False,
            errors=[str(exc)],
            warnings=[],
            counts={},
            source="unavailable",
        )
