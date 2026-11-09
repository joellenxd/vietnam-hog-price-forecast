"""P6: kiem thu dm_test.py (Diebold-Mariano + hieu chinh Harvey)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dm_test as dm  # noqa: E402


def series(x, start="2026-07-01"):
    return pd.Series(np.asarray(x, float), index=pd.date_range(start, periods=len(x), freq="D"))


class TestDMStatistic(unittest.TestCase):
    def test_h1_equals_one_sample_t_test(self):
        d = series(np.random.default_rng(1).normal(0.3, 1.0, 80))
        res = dm.dm_test(d, 1)
        t = stats.ttest_1samp(d.to_numpy(), 0.0)
        self.assertAlmostEqual(res["dm_stat"], t.statistic, places=9)
        self.assertAlmostEqual(res["p_value"], t.pvalue, places=9)
        self.assertEqual(res["n"], 80)

    def test_sign_convention_a_better_is_negative(self):
        rng = np.random.default_rng(2)
        ea, eb = rng.normal(0, 1, 150), rng.normal(0, 3, 150)
        d = series(np.abs(ea) - np.abs(eb))
        r = dm.dm_test(d, 7)
        self.assertLess(r["dm_stat"], 0)
        self.assertLess(r["p_value"], 0.01)
        r2 = dm.dm_test(-d, 7)
        self.assertAlmostEqual(r2["dm_stat"], -r["dm_stat"], places=9)
        self.assertAlmostEqual(r2["p_value"], r["p_value"], places=9)

    def test_identical_forecasts_give_nan(self):
        r = dm.dm_test(series(np.zeros(60)), 7)
        self.assertTrue(np.isnan(r["dm_stat"]) and np.isnan(r["p_value"]))

    def test_too_few_observations_give_nan(self):
        r = dm.dm_test(series(np.random.default_rng(3).normal(size=8)), 7)
        self.assertTrue(np.isnan(r["dm_stat"]))

    def test_handles_calendar_gaps(self):
        d = series(np.random.default_rng(4).normal(0.5, 1, 120))
        d = d.drop(d.index[::7])  # mat 1/7 ngay (nhu Chu nhat sao chep)
        r = dm.dm_test(d, 7)
        self.assertEqual(r["n"], len(d))
        self.assertTrue(np.isfinite(r["dm_stat"]))

    def test_size_control_with_overlapping_errors(self):
        """d_t la MA(h-1) khi gia thuyet H0 dung: DM co dung h khong qua nghiem, DM voi h=1 qua nghiem."""
        rng = np.random.default_rng(5)
        h, n, sims = 7, 200, 400
        rej_h = rej_1 = 0
        for _ in range(sims):
            e = rng.normal(size=n + h - 1)
            d = series(np.convolve(e, np.ones(h) / h, mode="valid"))
            rej_h += dm.dm_test(d, h)["p_value"] < 0.05
            rej_1 += dm.dm_test(d, 1)["p_value"] < 0.05
        self.assertLess(rej_h / sims, 0.12)
        self.assertGreater(rej_1 / sims, 0.25)

    def test_hln_is_more_conservative_than_plain_dm(self):
        d = series(np.random.default_rng(6).normal(0.4, 1, 70))
        self.assertLess(abs(dm.dm_test(d, 14, hln=True)["dm_stat"]), abs(dm.dm_test(d, 14, hln=False)["dm_stat"]))


def make_pred(model, err_scale, seed, h=7, n=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-07-01", periods=n, freq="D")
    rows = []
    for region in ("Bac", "Trung", "Nam"):
        y = 60000 + rng.normal(0, 500, n)
        p = y + rng.normal(0, err_scale, n)
        rows.append(pd.DataFrame({"model": model, "config": "c6", "region": region, "horizon": h,
                                  "origin_date": dates.strftime("%Y-%m-%d"),
                                  "target_date": (dates + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                                  "y_true": y, "y_pred": p}))
    return pd.concat(rows, ignore_index=True)


class TestPairwise(unittest.TestCase):
    def test_pairwise_table_shape_and_winner(self):
        preds = pd.concat([make_pred("good", 200, 1), make_pred("bad", 2000, 2)], ignore_index=True)
        preds["err"] = preds.y_true - preds.y_pred
        res = dm.pairwise_dm(preds)
        self.assertEqual(list(res.columns), dm.RESULT_COLUMNS)
        self.assertEqual(len(res), 1 * 2 * 4)  # 1 cap x 2 loss x (3 mien + all)
        row = res[(res.region == "all") & (res.loss == "ae")].iloc[0]
        self.assertEqual(row.better, "good")
        self.assertLess(row.p_value, 0.01)

    def test_only_common_origins_are_compared(self):
        a, b = make_pred("a", 300, 1), make_pred("b", 300, 2)
        b = b[b.origin_date >= "2026-07-31"]
        preds = pd.concat([a, b], ignore_index=True)
        preds["err"] = preds.y_true - preds.y_pred
        res = dm.pairwise_dm(preds)
        self.assertTrue((res[res.region == "Bac"].n == 30).all())

    def test_real_files_naive_vs_arima(self):
        files = [ROOT / "predictions" / "naive_c6.csv", ROOT / "predictions" / "p3_arima_sarimax.csv"]
        res = dm.pairwise_dm(dm.load_predictions(files))
        # 3 cap (arima-naive, naive-sarimax, arima-sarimax) x 3 horizon x 2 loss x 4 region
        self.assertEqual(len(res), 3 * 3 * 2 * 4)
        bac7 = res[(res.model_a == "arima") & (res.model_b == "naive") & (res.region == "Bac") & (res.horizon == 7)]
        self.assertTrue(bac7.dm_stat.isna().all())  # ARIMA(0,1,0) o mien Bac trung Naive
        self.assertTrue(res.p_value.dropna().between(0, 1).all())

    def test_duplicate_predictions_rejected(self):
        p = ROOT / "predictions" / "naive_c6.csv"
        with self.assertRaises(ValueError):
            dm.load_predictions([p, p])


if __name__ == "__main__":
    unittest.main()
