"""Contract checks for shared preprocessing and participant 3's saved split."""
import unittest
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import credit_pipeline as cp


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.test = cp.load_data()
        cls.tr, cls.va = cp.load_split()

    def test_saved_split_matches_contract(self):
        tr, va = cp.make_split(self.y)
        self.assertTrue(tr.equals(self.tr) and va.equals(self.va))
        self.assertFalse(len(tr.intersection(va)))
        self.assertEqual(set(tr) | set(va), set(self.X.index))
        self.assertEqual(set(self.y.loc[tr]), {0, 1})
        self.assertEqual(set(self.y.loc[va]), {0, 1})

    def test_transform_does_not_learn_from_validation(self):
        pre = cp.build_preprocessor().fit(self.X.loc[self.tr])
        money = pre['columns'].named_transformers_['money']
        learned = money[0].statistics_.copy()
        clean = cp.DomainCleaner().transform(self.X.loc[self.tr])
        np.testing.assert_allclose(learned, clean[cp.MONEY_COLS].median().to_numpy())
        altered = self.X.loc[self.va].copy()
        altered['Annual Income'] = 1e12
        pre.transform(altered)
        np.testing.assert_array_equal(learned, money[0].statistics_)

    def test_unknown_categories_missing_values_and_row_order(self):
        pre = cp.build_preprocessor().fit(self.X.loc[self.tr])
        probe = self.test.iloc[[7, 2, 5]].copy()
        original = probe.copy(deep=True)
        probe.loc[probe.index[0], 'Home Ownership'] = 'unseen home'
        probe.loc[probe.index[0], 'Years in current job'] = 'unseen tenure'
        probe.loc[probe.index[1], 'Purpose'] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # unknown-safe encoding may warn
            z = pre.transform(probe)
            per_row = np.vstack([pre.transform(probe.loc[[i]]) for i in probe.index])
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z, per_row)
        self.assertEqual(cp.DomainCleaner().transform(probe).iloc[0]['tenure_missing'], 1)
        self.assertTrue(self.test.iloc[[7, 2, 5]].equals(original))
        ztest = pre.transform(self.test)
        self.assertEqual(len(ztest), 750)
        self.assertNotIn(cp.TARGET, self.test)

    def test_score_hypotheses_and_no_mutation(self):
        row = self.X.iloc[[0]].copy()
        row['Credit Score'] = 7000.0
        row['Current Loan Amount'] = float(cp.LOAN_SENTINEL)
        original = row.copy(deep=True)
        self.assertEqual(cp.DomainCleaner(score_policy='divide').transform(row)['Credit Score'].iloc[0], 700)
        self.assertTrue(cp.DomainCleaner(score_policy='missing').transform(row)['Credit Score'].isna().all())
        self.assertEqual(cp.DomainCleaner(score_policy='raw').transform(row)['Credit Score'].iloc[0], 7000)
        self.assertTrue(cp.DomainCleaner().transform(row)['Current Loan Amount'].isna().all())
        pd.testing.assert_frame_equal(row, original)
        with self.assertRaises(ValueError):
            cp.DomainCleaner(score_policy='invalid').transform(row)

    def test_feature_switches_work_for_other_models(self):
        for flags in [False, True]:
            for scale in [False, True]:
                pre = cp.build_preprocessor(use_anomaly_flags=flags, scale=scale,
                                            min_frequency=10, score_policy='missing')
                z = pre.fit_transform(self.X.loc[self.tr])
                names = pre['columns'].get_feature_names_out()
                self.assertEqual('score_scale_flag' in names, flags)
                self.assertEqual(z.shape[1], len(names))
                self.assertTrue(np.isfinite(z).all())


if __name__ == '__main__':
    unittest.main()
