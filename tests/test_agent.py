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
        plan=agent.schedule('2026-01-31','2026-02-28')
        self.assertEqual(len(plan),2784)
        self.assertTrue((plan.availability_bound_utc<=plan.issue_utc).all())
        self.assertEqual(list(agent.safe_offset([1,12,13,36,37,48])),[1,1,2,2,3,3])
        self.assertFalse(plan.duplicated(['issue_utc','valid_utc','turbine_id']).any())

    def test_february_has_one_day_ahead_prediction_per_hour(self):
        plan=agent.schedule('2026-01-31','2026-02-28')
        feb=plan[(plan.lead_hours<=24)&(plan.valid_local<'2026-03-01')]
        self.assertEqual(len(feb),1344)
        self.assertEqual(feb.groupby('turbine_id').size().to_dict(),{1:672,2:672})
        self.assertFalse(feb.duplicated(['valid_local','turbine_id']).any())

    def test_scada_offset_moves_utc_and_rejects_stale_model(self):
        try:
            agent.set_scada_offset(6)
            plan=agent.schedule('2026-01-31','2026-01-31')
            self.assertEqual(str(plan.issue_utc.iloc[0]),'2026-01-31 17:00:00+00:00')
            weather=plan[['valid_utc','valid_local','turbine_id','offset_days']].copy()
            for col in ['wind','temp','direction']: weather[col]=np.nan
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/'models').mkdir()
                bundle={'scada_utc_offset_hours':5}
                with patch.object(agent,'ROOT',root),patch.object(agent,'read_weather',return_value=weather),patch.object(agent.joblib,'load',return_value=bundle),self.assertRaisesRegex(ValueError,'Model trained'):
                    agent.replay('2026-01-31','2026-01-31')
        finally:
            agent.set_scada_offset(5)

    def test_timezone_audit_does_not_claim_utc_from_naive_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for tid in [1,2]:
                times=pd.date_range('2024-02-29',periods=288,freq='10min')
                pd.DataFrame({'id':range(288),'time':times,'wind':5.,'power':.3,'temp':10.}).to_csv(root/f'turbine {tid}.csv',index=False)
            with patch.object(agent,'ROOT',root): report=agent.timezone_audit(root)
            self.assertEqual(report['inference'],'indeterminate_from_naive_csv')
            self.assertEqual(report['turbines']['1']['transition_day_counts'],{'2024-02-29':144,'2024-03-01':144})

    def test_hourly_target_requires_all_six_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for tid in [1,2]:
                times=pd.date_range('2025-01-01',periods=11,freq='10min')
                pd.DataFrame({'id':range(11),'time':times,'wind':5.,'power':.3,'temp':10.}).to_csv(root/f'turbine {tid}.csv',index=False)
            with patch.object(agent,'ROOT',root): frame=agent.prepare(root)
            self.assertEqual(frame.power.notna().sum(),2)
            self.assertEqual(frame.power.isna().sum(),2)
            self.assertTrue(np.allclose(frame.power.dropna(),.3))

    def test_duplicate_scada_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame([[1,'2025-01-01',4,.2,10],[2,'2025-01-01',4,.2,10]],columns=['id','time','wind','power','temp']).to_csv(root/'turbine 1.csv',index=False)
            with patch.object(agent,'ROOT',root),self.assertRaisesRegex(ValueError,'Duplicate'):
                agent.prepare(root)

    def test_no_actual_power_or_scada_wind_in_features(self):
        frame=agent.schedule('2026-01-31','2026-01-31')
        frame['wind']=5.;frame['temp']=10.;frame['direction']=90.;frame['power']=1.
        first=agent.features(frame)
        frame['power']=0.
        pd.testing.assert_frame_equal(first,agent.features(frame))
        self.assertNotIn('power',first.columns)

    def test_missing_weather_is_preserved_for_fallback(self):
        plan=agent.schedule('2026-01-31','2026-01-31')
        weather=plan[['valid_utc','valid_local','turbine_id','offset_days']].iloc[:1].copy()
        weather['wind']=5.;weather['temp']=10.;weather['direction']=90.
        merged=agent.forecast_frame(plan,weather)
        self.assertEqual(len(merged),96)
        self.assertEqual(merged.wind.isna().sum(),95)

    def test_agent_publishes_explicit_fallback_when_weather_missing(self):
        weather=agent.schedule('2026-01-31','2026-01-31')[['valid_utc','valid_local','turbine_id','offset_days']].copy()
        for col in ['wind','temp','direction']: weather[col]=np.nan
        bundle={'model':None,'means':{1:.3,2:.4},'radii':{'1_1':.2,'1_2':.2,'2_1':.2,'2_2':.2},'scada_utc_offset_hours':5}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(agent,'ROOT',root),patch.object(agent,'read_weather',return_value=weather),patch.object(agent.joblib,'load',return_value=bundle),patch.object(agent,'digest',return_value='test-model'):
                agent.replay('2026-01-31','2026-01-31')
            result=pd.read_csv(root/'results/cycles/2026-01-31/forecasts_48h.csv')
            self.assertEqual(len(result),96)
            self.assertTrue((result.status=='fallback_climatology').all())
            self.assertTrue((result.lower90==0).all())
            self.assertTrue((result.upper90==1).all())
            self.assertFalse((root/'results/forecasts_48h.csv').exists())

if __name__=='__main__': unittest.main()
