"""P6: kiem thu STL-PCA-BiLSTM (tinh nhan qua, chong ro ri, dinh dang). Chay: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stl_pca as sp  # noqa: E402
from evaluate import validate_predictions  # noqa: E402


def synthetic_series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-07-02", periods=n, freq="D")
    y = 60000 + 40 * np.arange(n) + 800 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 300, n)
    return pd.Series(y, index=idx)


class TestCausalSTL(unittest.TestCase):
    def test_warmup_is_nan_then_finite(self):
        c = sp.rolling_stl(synthetic_series())
        self.assertTrue(c.iloc[: sp.STL_MIN_OBS - 1].isna().all().all())
        self.assertTrue(np.isfinite(c.iloc[sp.STL_MIN_OBS - 1:].to_numpy()).all())

    def test_future_values_do_not_change_past_components(self):
        s = synthetic_series()
        base = sp.rolling_stl(s)
        s2 = s.copy()
        s2.iloc[80:] = s2.iloc[80:] * 5 + 10000  # pha hong tuong lai tu ngay 80
        alt = sp.rolling_stl(s2)
        np.testing.assert_allclose(base.iloc[:80].to_numpy(), alt.iloc[:80].to_numpy(), equal_nan=True)
        self.assertFalse(np.allclose(base.iloc[85:].to_numpy(), alt.iloc[85:].to_numpy()))

    def test_components_sum_to_price(self):
        s = synthetic_series()
        c = sp.rolling_stl(s)
        ok = c.notna().all(axis=1)
        np.testing.assert_allclose(c.loc[ok].sum(axis=1), s.loc[ok], rtol=1e-6)

    def test_interior_gap_uses_only_past(self):
        s = synthetic_series()
        s.iloc[50:53] = np.nan
        c = sp.rolling_stl(s)
        self.assertTrue(np.isfinite(c.iloc[53:].to_numpy()).all())


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = sp.add_stl_features(sp.load_data())
        cls.exog = sp.exog_columns()

    def test_assert_no_leak_passes(self):
        sp.assert_no_leak(self.df)

    def test_pca_ignores_val_and_test(self):
        a = sp.ExogPCA(self.exog, 0.95).fit(self.df)
        mod = self.df.copy()
        m = mod["split"] != "train"
        mod.loc[m, self.exog] = 123.0
        b = sp.ExogPCA(self.exog, 0.95).fit(mod)
        np.testing.assert_allclose(a.pca.components_, b.pca.components_)
        np.testing.assert_allclose(a.scaler.mean_, b.scaler.mean_)

    def test_exog_columns_exclude_price_and_lag90(self):
        self.assertFalse(any(c.endswith("_lag90") for c in self.exog))
        for c in ("price_t", "lag_7", "region_code", "y_h7"):
            self.assertNotIn(c, self.exog)

    def test_window_features_ignore_future_components(self):
        pipe = sp.Pipeline(self.df, self.exog, 14, 0.90)
        df2 = self.df.copy()
        cut = pd.Timestamp("2026-07-15")
        df2.loc[df2["date"] > cut, sp.COMP] = 1e6  # tuong lai cua ngay cat bi pha hong
        pipe2 = sp.Pipeline(df2, self.exog, 14, 0.90)
        d = self.df[(self.df["date"] == cut) & (self.df["region"] == "Bac")]
        r1 = pipe.win.rows(d["region"], d["date"])[0]
        r2 = pipe2.win.rows(d["region"], d["date"])[0]
        self.assertGreaterEqual(r1, 0)
        np.testing.assert_allclose(pipe.win.x[r1], pipe2.win.x[r2])

    def test_training_rows_have_no_unrevealed_label(self):
        cfg = sp.load_config(sp.CONFIG_PATH)
        pipe = sp.Pipeline(self.df, self.exog, 14, 0.90)
        from walkforward import available_training_rows
        for h in sp.HORIZONS:
            for w in sp.get_walkforward_windows(cfg, h):
                tr = available_training_rows(pipe.df, w["retrain_date"], h)
                d, *_ = pipe.design(tr, h)
                self.assertTrue(((d["date"] + pd.Timedelta(days=h)) < pd.Timestamp(w["retrain_date"])).all())

    def test_eval_origins_match_p3_naive(self):
        cfg = sp.load_config(sp.CONFIG_PATH)
        naive = pd.read_csv(ROOT / "predictions" / "naive_c6.csv")
        for h in sp.HORIZONS:
            rows = sp.eval_rows(self.df.loc[self.df["split"] == "test"], h, cfg["test"]["start"], cfg["test"]["end"])
            want = naive[naive.horizon == h].groupby("region").size()
            got = rows.groupby("region").size()
            pd.testing.assert_series_equal(got.sort_index(), want.sort_index(), check_names=False, check_dtype=False)


class TestModel(unittest.TestCase):
    def test_seed_reproducible(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 14, 8)).astype(np.float32)
        r = rng.integers(0, 3, 40)
        y = rng.normal(size=40)
        p = {"hidden": 8, "dropout": 0.1, "lr": 3e-3}
        a = sp.predict(sp.train_model(x, r, y, p, 42, 3), x, r)
        b = sp.predict(sp.train_model(x, r, y, p, 42, 3), x, r)
        c = sp.predict(sp.train_model(x, r, y, p, 7, 3), x, r)
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))


class TestPredictionFile(unittest.TestCase):
    def test_file_schema_if_present(self):
        f = ROOT / "predictions" / "stl_pca_bilstm_c6.csv"
        if not f.exists():
            self.skipTest("chua chay stl_pca.py")
        d = validate_predictions(pd.read_csv(f))
        self.assertEqual(set(d.model), {"stl_pca_bilstm"})
        self.assertEqual(set(d.config), {"c6"})
        self.assertTrue((d.origin_date >= "2026-07-01").all() and (d.target_date <= "2026-10-30").all())
        self.assertTrue(np.isfinite(d.y_pred).all())


if __name__ == "__main__":
    unittest.main()
