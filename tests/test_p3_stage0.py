import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluate import evaluate,validate_predictions,evaluate_predictions
from walkforward import load_config,retrain_dates,segment_origins,available_training_rows,get_walkforward_windows
from naive_arima import run_naive

class TestEvaluate(unittest.TestCase):
 def test_known_metrics(self):
  d=evaluate([10,20],[8,22]);self.assertEqual(d['n'],2);self.assertAlmostEqual(d['mae'],2);self.assertAlmostEqual(d['rmse'],2);self.assertAlmostEqual(d['mape'],15)
 def test_missing_pair(self):self.assertEqual(evaluate([10,np.nan,20],[11,15,19])['n'],2)
 def test_zero_mape(self):self.assertAlmostEqual(evaluate([0,10],[3,9])['mape'],10)
 def test_empty_error(self):
  with self.assertRaises(ValueError):evaluate([np.nan],[1])
 def test_schema_rejects_date_mismatch(self):
  d=pd.DataFrame([dict(model='naive',config='c6',region='Bac',horizon=7,origin_date='2026-07-01',target_date='2026-07-09',y_true=1,y_pred=1)])
  with self.assertRaises(ValueError):validate_predictions(d)
 def test_sunday_exclusion(self):
  d=pd.DataFrame([dict(model='naive',config='c6',region='Bac',horizon=7,origin_date=x,target_date=(pd.Timestamp(x)+pd.Timedelta(days=7)).date().isoformat(),y_true=10,y_pred=8) for x in ['2026-07-01','2026-07-02']])
  q=pd.DataFrame({'date':['2026-07-01','2026-07-02'],'region':['Bac','Bac'],'is_sunday_copy':[1,0]})
  self.assertEqual(evaluate_predictions(d,q).iloc[0]['n'],1)
 def test_duplicate_rejected(self):
  d=pd.DataFrame([dict(model='naive',config='c6',region='Bac',horizon=7,origin_date='2026-07-01',target_date='2026-07-08',y_true=1,y_pred=1)]*2)
  with self.assertRaises(ValueError):validate_predictions(d)

class TestWalkForward(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=load_config(ROOT/'data/split/split_config.json')
 def test_four_dates(self):self.assertEqual([x.strftime('%Y-%m-%d') for x in retrain_dates(self.cfg)],['2026-07-01','2026-08-01','2026-09-01','2026-10-01'])
 def test_month_map(self):self.assertEqual(segment_origins(['2026-07-31','2026-08-01'],self.cfg).dt.strftime('%Y-%m-%d').tolist(),['2026-07-01','2026-08-01'])
 def test_future_leak_prevention(self):
  d=pd.DataFrame({'date':['2026-06-22','2026-06-23','2026-07-01']})
  self.assertEqual(available_training_rows(d,'2026-07-01',7)['date'].tolist(),['2026-06-22','2026-06-23'])
 def test_windows(self):self.assertEqual(get_walkforward_windows(self.cfg,30)[-1]['prediction_end'],'2026-10-30')

class TestNaive(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=pd.read_csv(ROOT/'data/raw/csv/supervised_region_daily.csv')
  cls.pred=run_naive(cls.raw,load_config(ROOT/'data/split/split_config.json'))
 def test_9_groups(self):self.assertEqual(len(self.pred.groupby(['region','horizon'])),9)
 def test_pred_equals_origin_price(self):
  a=self.pred.merge(self.raw[['date','region','price_ffill']],left_on=['origin_date','region'],right_on=['date','region'],validate='many_to_one')
  self.assertTrue(np.allclose(a.y_pred,a.price_ffill))
 def test_only_test_origins(self):self.assertTrue((pd.to_datetime(self.pred.origin_date)>=pd.Timestamp('2026-07-01')).all())
 def test_test_end_horizon(self):self.assertTrue((pd.to_datetime(self.pred.target_date)<=pd.Timestamp('2026-10-30')).all())
 def test_no_origin_sundays(self):
  a=self.pred.merge(self.raw[['date','region','is_sunday_copy']],left_on=['origin_date','region'],right_on=['date','region'])
  self.assertTrue(a.is_sunday_copy.eq(0).all())
 def test_no_missing_target(self):self.assertTrue(np.isfinite(self.pred.y_true).all())

if __name__=='__main__':unittest.main()
