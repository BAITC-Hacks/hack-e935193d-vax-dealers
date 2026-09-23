# Vax-Dealers: запуск прогноза ВЭС

## Что потребуется

- Python 3.12 и доступ к интернету для установки зависимостей и загрузки архива погоды.
- Два исходных CSV в папке `raw/`. Имена должны оканчиваться на `turbine 1.csv` и `turbine 2.csv`. Программа ищет по одному файлу для каждой турбины.
- Команды ниже выполняются из корня репозитория. API-ключ для Open-Meteo не требуется.

Готовые результаты можно посмотреть без запуска программы: скачайте [архив CSV](results/forecast_results.zip) и распакуйте его в корень проекта.

## Установка

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Полный расчёт

```bash
python wind_agent.py prepare --data-dir raw
python wind_agent.py fetch
python wind_agent.py train
python wind_agent.py replay
```

1. `prepare` проверяет CSV, собирает полные часы из шести десятиминутных записей и создаёт `data/hourly.csv`, `reports/data_audit.json` и `reports/timezone_audit.json`.
2. `fetch` загружает архивные прогнозы GFS через Open-Meteo по координатам обеих турбин в `cache/`. По умолчанию берётся период с 01.04.2024 по 02.03.2026; уже скачанные месяцы используются повторно.
3. `train` обучает модель и создаёт `models/forecast.joblib`, `reports/model_card.json`, `reports/january_metrics.csv` и `reports/january_predictions.csv`.
4. `replay` повторно получает погоду для расчётного интервала и формирует ежедневные 48-часовые прогнозы. По умолчанию даты выпусков — с 31.01 по 28.02.2026.

Для повторного расчёта без сетевых запросов после `fetch` используйте:

```bash
python wind_agent.py replay --offline
```

Если нужно принудительно обновить весь погодный кэш:

```bash
python wind_agent.py fetch --refresh
```

## Один выпуск и автоматическое обновление

Для одного исторического выпуска укажите одинаковые начальную и конечную даты:

```bash
python wind_agent.py replay --first-issue 2026-02-10 --last-issue 2026-02-10
```

Чтобы повторять загрузку погоды и расчёт для **этой же даты** каждый час, запустите:

```bash
python wind_agent.py watch --issue-date 2026-02-10 --interval-seconds 3600
```

`watch` работает до остановки Ctrl+C. Интервал должен быть не меньше 60 секунд. Дата выпуска зафиксирована параметром `--issue-date`: процесс сам не переходит к следующему дню. Для автоматического запуска после перезагрузки добавьте эту команду в Планировщик заданий Windows или в cron/systemd с рабочей папкой репозитория и Python из `.venv`.

Если нужна последовательная обработка **всего февраля**, используйте одну команду `python wind_agent.py replay`; она создаёт 29 ежедневных выпусков за один запуск. Для отдельного выпуска результаты записываются в `results/cycles/<дата>/`, поэтому он не заменяет таблицы полного расчёта.

## Где находятся результаты

- `results/forecasts_48h.csv` — все ежедневные выпуски, по 48 часов для каждой турбины. Поля `issue_utc`, `valid_utc` и `valid_local` показывают время выпуска и прогнозируемый час; `prediction` — нормализованная мощность от 0 до 1.
- `results/submission_february.csv` — по одному прогнозу на каждый час февраля для каждой турбины, выбранному из горизонта 1–24 часа.
- `results/station_february.csv` — оба прогноза рядом; столбец `equal_capacity_mean_assumption` предполагает равную номинальную мощность турбин.
- `results/run_summary.json` — число выпусков, строк и случаев резервного расчёта.
- `results/agent_events.jsonl` — журнал этапов, проверок и отпечатков входных данных. Каждое повторение дописывает события в этот файл.
- `reports/january_metrics.csv` — проверка модели на январских данных.

`status=weather_model` означает расчёт по погоде. `status=fallback_climatology` означает, что погодных данных не хватило и использовано историческое среднее; такой результат нужно проверять отдельно.

## Проверка и настройка времени

Запуск тестов:

```bash
python -m unittest discover -s tests -v
```

CSV не содержат часового пояса. По умолчанию программа считает их метки временем UTC+5. Для проверки признаков времени отдельно запустите:

```bash
python wind_agent.py timezone-audit --data-dir raw
```

Если оператор подтвердит другое смещение, укажите его **перед** командой во всём цикле подготовки, обучения и расчёта, например:

```bash
python wind_agent.py --scada-utc-offset-hours 6 prepare --data-dir raw
python wind_agent.py --scada-utc-offset-hours 6 fetch
python wind_agent.py --scada-utc-offset-hours 6 train
python wind_agent.py --scada-utc-offset-hours 6 replay
```

Модель хранит выбранное смещение и отклоняет расчёт с другим значением. В исходных CSV нет фактической выработки за февраль, поэтому февральская ошибка в результатах не вычисляется. Значения `prediction` нормализованы; для перевода в МВт и МВт·ч нужны номинальные мощности турбин.
