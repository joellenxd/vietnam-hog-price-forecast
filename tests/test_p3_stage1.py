import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arima_sarimax import known_calendar,prepare_region,run_block,run_experiment
from evaluate import validate_predictions
ROOT=Path(__file__).resolve().parents[1]

class TestStage1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(ROOT/'data/features/data_v1.csv')
        cls.pred=pd.read_csv(ROOT/'predictions/p3_arima_sarimax.csv')
        cls.metrics=pd.read_csv(ROOT/'results/tables/p3_arima_sarimax_metrics.csv')
        cls.select=pd.read_csv(ROOT/'results/tables/p3_arima_sarimax_val_selection.csv')
    def test_calendar_known_and_finite(self):
        z=known_calendar(pd.date_range('2026-01-01',periods=14));self.assertEqual(z.shape,(14,2));self.assertTrue(np.isfinite(z).all())
    def test_predictions_schema(self):
        self.assertEqual(len(validate_predictions(self.pred)),1626)
    def test_models_regions_horizons(self):
        self.assertEqual(set(self.pred.model),{'arima','sarimax'})
        self.assertEqual(set(self.pred.region),{'Bac','Trung','Nam'})
        self.assertEqual(set(self.pred.horizon),{7,14,30})
    def test_expected_counts(self):
        for (_,_,h),g in self.pred.groupby(['model','region','horizon']):
            self.assertEqual(len(g),{7:99,14:93,30:79}[h])
    def test_no_test_boundary_crossing(self):
        self.assertTrue((pd.to_datetime(self.pred.origin_date)>=pd.Timestamp('2026-07-01')).all())
        self.assertTrue((pd.to_datetime(self.pred.target_date)<=pd.Timestamp('2026-10-30')).all())
    def test_selected_val_configs(self):
        self.assertEqual(len(self.select),12)
        self.assertEqual(int(self.select.selected.sum()),6)
        self.assertEqual(len(self.metrics),18)
    def test_no_sunday_origins(self):
        idx=self.data.set_index(['date','region'])
        for r in self.pred.itertuples(index=False):
            self.assertEqual(int(idx.loc[(r.origin_date,r.region),'is_sunday_copy']),0)
    def test_actual_labels_match(self):
        idx=self.data.set_index(['date','region'])
        for r in self.pred.itertuples(index=False):
            self.assertAlmostEqual(float(idx.loc[(r.origin_date,r.region),f'y_h{r.horizon}']),r.y_true,places=4)

if __name__=='__main__':unittest.main()
