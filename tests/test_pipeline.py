"""Runs main.ipynb with an extra cell of section-2 checks appended."""
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CHECKS = """
assert "Id" not in X.columns and "SalePrice" not in X.columns
assert len(X) == len(train) and X.index.equals(y.index)
assert "MSSubClass" in categorical_features
assert X[absence_means_none].isna().sum().sum() == 0
# Learned preprocessing must live inside each pipeline, not be applied to X beforehand.
assert all(m.steps[0][0] == "preprocess" for m in models.values())
assert cv.shuffle and cv.random_state == 42
assert len(test) == original_test_row_count
best = cv_results["RMSE_mean"].idxmin()
assert cv_results.loc[best, "RMSE_mean"] < cv_results.loc["Dummy (median)", "RMSE_mean"]
assert cv_results.loc["Gradient Boosting", "R2_mean"] > 0.8
"""


def test_notebook_section_2(tmp_path):
    for name in ["main.ipynb", "train.csv", "test.csv", "data_description.txt"]:
        shutil.copy(ROOT / name, tmp_path / name)
    nb = nbformat.read(tmp_path / "main.ipynb", as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(CHECKS))
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(tmp_path)}}).execute()
