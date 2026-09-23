"""Reproducible weather-driven wind forecasting with an auditable policy agent."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import joblib

ROOT = Path(__file__).resolve().parent
COORDS = {1: (43.645150, 78.535604), 2: (43.643198, 78.538828)}
VARIABLES = ['wind_speed_80m', 'wind_direction_80m', 'temperature_2m']
FEATURES = ['wind', 'temp', 'direction_sin', 'direction_cos', 'hour_sin',
            'hour_cos', 'month_sin', 'month_cos', 'offset_days', 'turbine_id']
SCADA_UTC_OFFSET_HOURS = 5  # 2026 civil time; SCADA clock origin cannot be proven from naive CSV timestamps.
OFFSET = pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)
LATENCY_HOURS = 12

def set_scada_offset(hours):
    """Use one explicitly chosen SCADA clock offset for all UTC/forecast joins."""
    global SCADA_UTC_OFFSET_HOURS, OFFSET
    if not isinstance(hours, int) or not -12 <= hours <= 14:
        raise ValueError('SCADA UTC offset must be an integer between -12 and +14 hours')
    SCADA_UTC_OFFSET_HOURS = hours
    OFFSET = pd.Timedelta(hours=hours)

def timezone_audit(data_dir):
    """Record evidence about CSV wall-clock labels without claiming an unrecorded UTC offset."""
    report = {
        'inference': 'indeterminate_from_naive_csv',
        'civil_time_2026': 'Asia/Almaty (UTC+05:00)',
        'civil_time_source': 'https://primeminister.kz/ru/decisions/19012024-20',
        'assumed_scada_utc_offset_hours': SCADA_UTC_OFFSET_HOURS,
        'warning': 'The files contain no UTC offset or independently synchronized timestamp. '
                   'Local civil time does not establish the controller clock or the interval-label convention.',
        'turbines': {},
    }
    for tid in COORDS:
        matches = list(Path(data_dir).glob(f'*turbine {tid}.csv'))
        if len(matches) != 1:
            raise ValueError(f'Expected one turbine {tid}.csv, got {matches}')
        raw = pd.read_csv(matches[0], usecols=[1, 4])
        raw.columns = ['time', 'temp']
        raw['time'] = pd.to_datetime(raw.time, errors='raise')
        transition = raw[(raw.time >= '2024-02-29') & (raw.time < '2024-03-02')]
        counts = transition.groupby(transition.time.dt.strftime('%Y-%m-%d')).size().to_dict()
        winter_hours = {}
        for label, start, end in [('winter_2023_24', '2023-12-01', '2024-03-01'),
                                  ('winter_2024_25', '2024-12-01', '2025-03-01'),
                                  ('winter_2025_26', '2025-12-01', '2026-02-01')]:
            winter = raw[(raw.time >= start) & (raw.time < end)]
            hourly = winter.groupby(winter.time.dt.hour).temp.mean()
            if len(hourly) == 24:
                winter_hours[label] = dict(coldest_hour=int(hourly.idxmin()),
                                           warmest_hour=int(hourly.idxmax()))
        report['turbines'][str(tid)] = dict(source=matches[0].name,
            transition_day_counts=counts, transition_duplicate_labels=int(transition.time.duplicated().sum()),
            winter_temperature_clock_hours=winter_hours)
    dump(ROOT/'reports/timezone_audit.json', report)
    return report

def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fetch_month(tid, start, end, refresh=False):
    path = ROOT/'cache'/f'gfs_t{tid}_{start}_{end}.json'
    if path.exists() and not refresh:
        return path
    lat, lon = COORDS[tid]
    query = dict(latitude=lat, longitude=lon, start_date=str(start), end_date=str(end),
                 hourly=','.join(f'{v}_previous_day{d}' for d in [1,2,3] for v in VARIABLES),
                 models='gfs_seamless', timezone='GMT', wind_speed_unit='ms')
    url = 'https://previous-runs-api.open-meteo.com/v1/forecast?' + urllib.parse.urlencode(query)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            data = json.loads(raw)
            if 'hourly' not in data: raise ValueError('Missing hourly weather')
            data['_provenance'] = dict(url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
                sha256_payload=hashlib.sha256(raw).hexdigest(), lat=lat, lon=lon,
                provider='Open-Meteo Previous Runs / NOAA GFS')
            dump(path, data)
            return path
        except Exception:
            if attempt == 3: raise
            time.sleep(2**attempt)

def fetch_range(start, end, refresh=False):
    day = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while day <= end:
        last = min(day + pd.offsets.MonthEnd(0), end)
        for tid in COORDS:
            print('Weather', fetch_month(tid, day.date(), last.date(), refresh), flush=True)
        day = last + pd.Timedelta(days=1)

def prepare(data_dir):
    frames, audit = [], {}
    for tid in COORDS:
        matches = list(Path(data_dir).glob(f'*turbine {tid}.csv'))
        if len(matches) != 1: raise ValueError(f'Expected one turbine {tid}.csv, got {matches}')
        path = matches[0]
        raw = pd.read_csv(path)
        if len(raw.columns) != 5: raise ValueError('Unexpected SCADA schema')
        raw.columns = ['id', 'time', 'wind', 'power', 'temp']
        raw['time'] = pd.to_datetime(raw.time, errors='raise')
        if raw.time.duplicated().any(): raise ValueError('Duplicate SCADA timestamps')
        for col in ['power', 'wind', 'temp']:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        bad = (~raw.power.between(0,1) | ~raw.wind.between(0,80) | ~raw.temp.between(-70,70))
        off_grid = (raw.time.dt.minute % 10 != 0) | (raw.time.dt.second != 0)
        bad |= off_grid
        clean = raw.loc[~bad].set_index('time').sort_index()
        hour = clean[['power', 'wind', 'temp']].resample('h').mean()
        hour['n_samples'] = clean.power.resample('h').count()
        # No interpolation of target. Every accepted hour has six ten-minute samples.
        hour.loc[hour.n_samples != 6, ['power', 'wind', 'temp']] = np.nan
        hour['turbine_id'] = tid
        hour.index.name = 'valid_local'
        frames.append(hour.reset_index())
        audit[tid] = dict(file=path.name, sha256=digest(path), rows=len(raw),
            start=str(raw.time.min()), end=str(raw.time.max()), invalid_rows=int(bad.sum()),
            total_hours=len(hour), complete_hours=int((hour.n_samples == 6).sum()),
            incomplete_hours=int((hour.n_samples != 6).sum()),
            february_rows=int((raw.time >= '2026-02-01').sum()),
            missing_10min_slots=int(len(pd.date_range(raw.time.min(), raw.time.max(), freq='10min'))-len(raw)))
    data = pd.concat(frames, ignore_index=True)
    (ROOT/'data').mkdir(exist_ok=True)
    data.to_csv(ROOT/'data/hourly.csv', index=False)
    dump(ROOT/'reports/data_audit.json', audit)
    return data

def read_weather():
    records = []
    snapshots = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in (ROOT/'cache').glob('gfs_t*.json')]
    snapshots.sort(key=lambda item: item[1].get('_provenance',{}).get('retrieved_at',''))
    for path, data in snapshots:
        tid = int(path.name.split('_')[1][1:])
        h = data['hourly']
        if data['hourly_units']['wind_speed_80m_previous_day1'] != 'm/s':
            raise ValueError('Wrong wind units')
        utc = pd.to_datetime(h['time'], utc=True)
        for days in [1,2,3]:
            records.append(pd.DataFrame(dict(valid_utc=utc, turbine_id=tid, offset_days=days,
                wind=h[f'wind_speed_80m_previous_day{days}'],
                direction=h[f'wind_direction_80m_previous_day{days}'],
                temp=h[f'temperature_2m_previous_day{days}'])))
    if not records: raise ValueError('No weather archive; run fetch first')
    df = pd.concat(records, ignore_index=True)
    df = df.drop_duplicates(['valid_utc','turbine_id','offset_days'], keep='last')
    df['valid_local'] = df.valid_utc.dt.tz_localize(None) + OFFSET
    return df

def features(frame):
    x = frame.copy()
    x['direction_sin'] = np.sin(np.deg2rad(x.direction))
    x['direction_cos'] = np.cos(np.deg2rad(x.direction))
    for name, values, period in [('hour', x.valid_local.dt.hour, 24), ('month', x.valid_local.dt.month, 12)]:
        x[name+'_sin'] = np.sin(2*np.pi*values/period)
        x[name+'_cos'] = np.cos(2*np.pi*values/period)
    return x[FEATURES].astype(float)

def safe_offset(lead_hours):
    """Use weather with at least 12 hours between nominal init and issue."""
    return np.ceil((np.asarray(lead_hours) + LATENCY_HOURS)/24).astype(int)

def schedule(first_issue, last_issue):
    rows = []
    for day in pd.date_range(first_issue, last_issue, freq='D'):
        issue_local = day.normalize() + pd.Timedelta(hours=23)
        issue_utc = (issue_local - OFFSET).tz_localize('UTC')
        for lead in range(1,49):
            valid = issue_utc + pd.Timedelta(hours=lead)
            days = int(safe_offset(lead))
            for tid in COORDS:
                rows.append(dict(issue_utc=issue_utc, valid_utc=valid,
                    valid_local=valid.tz_localize(None)+OFFSET, lead_hours=lead,
                    turbine_id=tid, offset_days=days,
                    availability_bound_utc=valid-pd.Timedelta(days=days)+pd.Timedelta(hours=LATENCY_HOURS)))
    frame = pd.DataFrame(rows)
    if not (frame.availability_bound_utc <= frame.issue_utc).all():
        raise ValueError('Future weather detected')
    return frame

def forecast_frame(plan, weather):
    return plan.merge(weather.drop(columns='valid_local'), on=['valid_utc','turbine_id','offset_days'],
                      how='left', validate='many_to_one')

def metrics(y, pred):
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = np.asarray(y)[valid], np.asarray(pred)[valid]
    if len(y)==0: return dict(n=0, MAE=None, RMSE=None, bias=None)
    return dict(n=len(y), MAE=float(np.mean(np.abs(y-pred))),
                RMSE=float(np.sqrt(np.mean((y-pred)**2))), bias=float(np.mean(pred-y)))

def train():
    scada = pd.read_csv(ROOT/'data/hourly.csv', parse_dates=['valid_local'])
    weather = read_weather()
    pairs = weather.merge(scada[['valid_local','turbine_id','power']], on=['valid_local','turbine_id'],
                          validate='many_to_one').dropna(subset=['power','wind','direction','temp'])
    train_rows = pairs[pairs.valid_local < '2025-12-01']
    dec = forecast_frame(schedule('2025-11-30','2025-12-30'), weather)
    dec = dec.merge(scada[['valid_local','turbine_id','power']], on=['valid_local','turbine_id'], how='left')
    dec = dec[(dec.valid_local < '2026-01-01')].dropna(subset=['power','wind','direction','temp'])
    if len(train_rows)<5000 or len(dec)<500: raise ValueError('Insufficient training/validation data')
    candidates = [(15, 180, 0.06), (31, 180, 0.06), (15, 300, 0.04)]
    scores = []
    with threadpool_limits(limits=2):
        for leaves, iterations, lr in candidates:
            model = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=iterations,
                learning_rate=lr, min_samples_leaf=40, l2_regularization=5,
                early_stopping=False, random_state=42)
            model.fit(features(train_rows), train_rows.power)
            score = metrics(dec.power, np.clip(model.predict(features(dec)),0,1))
            scores.append(dict(leaves=leaves, iterations=iterations, lr=lr, **score))
            print('December validation', scores[-1], flush=True)
        best = min(scores, key=lambda v: v['MAE'])
        final_rows = pairs[pairs.valid_local < '2026-01-01']
        model = HistGradientBoostingRegressor(max_leaf_nodes=best['leaves'], max_iter=best['iterations'],
            learning_rate=best['lr'], min_samples_leaf=40, l2_regularization=5,
            early_stopping=False, random_state=42).fit(features(final_rows), final_rows.power)
    jan = forecast_frame(schedule('2025-12-31','2026-01-30'), weather)
    jan = jan[jan.valid_local < '2026-02-01'].merge(scada[['valid_local','turbine_id','power']],
                                                  on=['valid_local','turbine_id'], how='left')
    jan['prediction'] = np.clip(model.predict(features(jan)),0,1)
    means = scada[scada.valid_local<'2026-01-01'].groupby('turbine_id').power.mean().to_dict()
    jan['climatology'] = jan.turbine_id.map(means)
    # Persistence uses only full hours completed before each historical issue.
    jan['persistence'] = np.nan
    for (issue,tid), part in jan.groupby(['issue_utc','turbine_id']):
        issue_local = issue.tz_localize(None)+OFFSET
        hist = scada[(scada.turbine_id==tid) & (scada.valid_local+pd.Timedelta(hours=1)<=issue_local)].dropna(subset=['power'])
        jan.loc[part.index,'persistence'] = hist.power.iloc[-1] if len(hist) else means[tid]
    results = []
    for tid in COORDS:
        for horizon in ['1-24','25-48']:
            part=jan[(jan.turbine_id==tid) & ((jan.lead_hours<=24) if horizon=='1-24' else (jan.lead_hours>24))]
            for method in ['prediction','climatology','persistence']:
                results.append(dict(turbine_id=tid,horizon=horizon,model=method,**metrics(part.power,part[method])))
    radii = {}
    for tid in COORDS:
        for h in [1,2]:
            part=jan[(jan.turbine_id==tid)&(((jan.lead_hours-1)//24+1)==h)
                     &(jan.valid_local < '2026-01-31 23:00')].dropna(subset=['power'])
            errors=np.abs(part.power-part.prediction)
            # Empirical temporal calibration; no exchangeability/coverage guarantee claimed.
            radii[f'{tid}_{h}']=float(np.quantile(errors,.9,method='higher'))
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(dict(model=model, means=means, radii=radii, features=FEATURES,
                     scada_utc_offset_hours=SCADA_UTC_OFFSET_HOURS), ROOT/'models/forecast.joblib')
    jan.to_csv(ROOT/'reports/january_predictions.csv',index=False)
    pd.DataFrame(results).to_csv(ROOT/'reports/january_metrics.csv',index=False)
    dump(ROOT/'reports/model_card.json',dict(selection=scores, selected=best, training_rows=len(final_rows),
        training_unique_hours=final_rows[['valid_local','turbine_id']].drop_duplicates().shape[0],
        train_target_end='2025-12-31 23:00 local', validation='2025-12', test_and_interval_calibration='2026-01',
        interval_radii=radii, features=FEATURES, target='mean normalized active power',
        timezone_assumption=f'UTC{SCADA_UTC_OFFSET_HOURS:+03d}:00', weather_latency_assumption_hours=LATENCY_HOURS,
        archive_provenance='Fixed lead offsets; publication timestamps not supplied by API'))
    print(pd.DataFrame(results).to_string(index=False), flush=True)

def replay(first_issue='2026-01-31', last_issue='2026-02-28', refresh=False):
    events=[]
    def event(state, **details):
        events.append(dict(recorded_at=datetime.now(timezone.utc).isoformat(),state=state,**details))
    event('PLAN', first_issue=first_issue,last_issue=last_issue)
    if refresh:
        event('FETCH')
        # Refresh only dates needed for this cycle; failures preserve cached data, flagged below.
        try:
            fetch_range(first_issue, pd.Timestamp(last_issue)+pd.Timedelta(days=3), refresh=True)
        except Exception as exc:
            event('FETCH_FAILED_USE_CACHE',error=str(exc))
    bundle=joblib.load(ROOT/'models/forecast.joblib')
    trained_offset = bundle.get('scada_utc_offset_hours', 5)
    if trained_offset != SCADA_UTC_OFFSET_HOURS:
        raise ValueError(f'Model trained with SCADA UTC offset {trained_offset:+d}; '
                         f'current offset is {SCADA_UTC_OFFSET_HOURS:+d}. Re-run prepare and train.')
    weather=read_weather()
    output=forecast_frame(schedule(first_issue,last_issue),weather)
    # Strict production replay refuses pre-training origins.
    if output.issue_utc.min() < pd.Timestamp('2026-01-31T18:00Z'):
        raise ValueError('Model includes January interval calibration; replay only from January 31')
    good=output[['wind','direction','temp']].notna().all(axis=1)
    good &= output.wind.between(0,80) & output.temp.between(-70,70) & output.direction.between(0,360)
    output['prediction']=output.turbine_id.map(bundle['means'])
    with threadpool_limits(limits=2):
        if good.any(): output.loc[good,'prediction']=np.clip(bundle['model'].predict(features(output[good])),0,1)
    output['status']=np.where(good,'weather_model','fallback_climatology')
    output['uncertainty_radius']=[bundle['radii'][f'{tid}_{(lead-1)//24+1}'] for tid,lead in zip(output.turbine_id,output.lead_hours)]
    output['lower90']=(output.prediction-output.uncertainty_radius).clip(0,1)
    output['upper90']=(output.prediction+output.uncertainty_radius).clip(0,1)
    output.loc[~good,['lower90','upper90']]=[0,1]
    output['ramp_flag']=False
    for _, part in output.groupby(['issue_utc','turbine_id']):
        output.loc[part.index,'ramp_flag']=part.prediction.diff().abs().gt(.3)
    model_hash=digest(ROOT/'models/forecast.joblib')
    for issue, part in output.groupby('issue_utc'):
        fingerprint=hashlib.sha256((model_hash+part[['valid_utc','turbine_id','wind','direction','temp','offset_days']].to_csv(index=False)).encode()).hexdigest()
        event('VALIDATE',issue_utc=str(issue),hours_per_turbine=48, missing_weather=int((part.status!='weather_model').sum()))
        event('PREDICT',issue_utc=str(issue),input_fingerprint=fingerprint,model_sha256=model_hash)
        event('REVIEW',issue_utc=str(issue),action='publish' if (part.status=='weather_model').all() else 'publish_degraded',
              min_prediction=float(part.prediction.min()), max_prediction=float(part.prediction.max()),
              ramp_flags=int(part.ramp_flag.sum()))
    if len(output)!=len(pd.date_range(first_issue,last_issue))*96: raise ValueError('Forecast coverage mismatch')
    if output.duplicated(['issue_utc','valid_utc','turbine_id']).any(): raise ValueError('Duplicate prediction keys')
    results_dir = ROOT/'results'
    if first_issue == last_issue:
        results_dir = results_dir/'cycles'/str(pd.Timestamp(first_issue).date())
    results_dir.mkdir(parents=True,exist_ok=True)
    output.to_csv(results_dir/'forecasts_48h.csv',index=False)
    # Exactly one day-ahead forecast per February hour, no overlapping-horizon double counting.
    submission=output[(output.lead_hours<=24)&(output.valid_local>='2026-02-01')&(output.valid_local<'2026-03-01')].copy()
    submission.to_csv(results_dir/'submission_february.csv',index=False)
    if len(submission):
        station=submission.pivot(index='valid_local',columns='turbine_id',values='prediction')
        station.columns=['turbine_1_normalized','turbine_2_normalized']
        station['equal_capacity_mean_assumption']=station.mean(axis=1)
        station.to_csv(results_dir/'station_february.csv')
    event('PUBLISH', rows=len(output),february_day_ahead_rows=len(submission),fallback_rows=int((~good).sum()))
    with (ROOT/'results/agent_events.jsonl').open('a',encoding='utf-8') as f:
        for item in events: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    dump(results_dir/'run_summary.json',dict(rows=len(output),february_rows=len(submission),
        fallback_rows=int((~good).sum()),issues=int(output.issue_utc.nunique()), model_sha256=model_hash,
        provenance_caveat='availability_bound_utc is an assumed conservative bound, not observed publication time'))
    print('Published',len(output),'forecasts; fallback rows:',int((~good).sum()),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scada-utc-offset-hours', type=int, default=5,
                   help='Assumed offset of naive CSV timestamps from UTC (default: +5, unverified)')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--data-dir',required=True)
    q=sub.add_parser('timezone-audit');q.add_argument('--data-dir',required=True)
    q=sub.add_parser('fetch');q.add_argument('--start',default='2024-04-01');q.add_argument('--end',default='2026-03-02');q.add_argument('--refresh',action='store_true')
    sub.add_parser('train')
    q=sub.add_parser('replay');q.add_argument('--first-issue',default='2026-01-31');q.add_argument('--last-issue',default='2026-02-28');q.add_argument('--offline',action='store_true',help='Use previously downloaded weather only')
    q=sub.add_parser('watch');q.add_argument('--issue-date',required=True);q.add_argument('--interval-seconds',type=int,default=3600)
    args=p.parse_args()
    set_scada_offset(args.scada_utc_offset_hours)
    if args.command=='prepare':
        prepare(args.data_dir)
        timezone_audit(args.data_dir)
    elif args.command=='timezone-audit': print(json.dumps(timezone_audit(args.data_dir),ensure_ascii=False,indent=2))
    elif args.command=='fetch': fetch_range(args.start,args.end,args.refresh)
    elif args.command=='train': train()
    elif args.command=='replay': replay(args.first_issue,args.last_issue,refresh=not args.offline)
    else:
        if args.interval_seconds<60: p.error('Minimum interval is 60 seconds')
        while True:
            replay(args.issue_date,args.issue_date,refresh=True)
            time.sleep(args.interval_seconds)

if __name__=='__main__': main()
