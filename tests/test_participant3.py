"""Checks for part 3 model helpers in credit_pipeline."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.impute import SimpleImputer
import credit_pipeline as cp


class Participant3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.test = cp.load_data()
        cls.tr, _ = cp.load_split()

    def test_class1_metrics(self):
        m = cp.class1_metrics([0, 1, 1, 0], [0, 1, 0, 1])
        self.assertEqual(m, {"precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(cp.class1_metrics([0, 1], [0, 0])["f1"], 0.0)

    def test_models_and_scaling(self):
        for flags in (True, False):
            self.assertEqual(set(cp.make_models(flags)), {"logreg", "random_forest", "hist_gb"})
        money = cp.make_models()["hist_gb"]["pre"]["columns"].transformers[0][1]
        self.assertIsInstance(money, SimpleImputer)

    def test_tune_threshold_fits_given_rows_and_predicts_test(self):
        rows = self.tr[:1500]
        tuned = cp.tune_threshold(cp.make_models()["hist_gb"], self.X.loc[rows], self.y.loc[rows])
        self.assertTrue(0 < tuned.best_threshold_ < 1)
        self.assertEqual(len(tuned.cv_results_["thresholds"]), 100)
        pred = tuned.predict(self.test)
        self.assertEqual(len(pred), 750)
        self.assertLessEqual(set(pred), {0, 1})


if __name__ == "__main__":
    unittest.main()
