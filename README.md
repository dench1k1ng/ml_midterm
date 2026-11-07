# ml_endterm — Project 5: Loan Approval (`Credit Default`)

Data: `Project_eng.zip` (`project-5/train_data.csv` — 6,750 rows, `project-5/test_data.csv` — 750 rows).
Test rows must not be deleted.

- Progress and next steps: [PROGRESS.md](PROGRESS.md)
- Guide for participant 2: [PARTICIPANT2_GUIDE.md](PARTICIPANT2_GUIDE.md)

| Part | Owner | Deliverable | Status |
|---|---|---|---|
| Data audit & EDA | Denis | `Denis_Credit_Default_EDA.ipynb` | done |
| Preprocessing, split, baseline | Participant 2 | `credit_pipeline.py`, `Participant2_Preprocessing_Baseline.ipynb`, `artifacts/participant2/` | done |
| Extra models, comparison, error analysis | Participant 3 | — | to do |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter lab   # open a notebook → Restart Kernel and Run All
```

Run notebooks from the repository root (they read `Project_eng.zip` and import `credit_pipeline.py`).

## For participant 3

```python
import credit_pipeline as cp
X_full, y_full, X_test = cp.load_data()        # test labels are NOT loaded
idx_tr, idx_va = cp.load_split()               # same 5,400 / 1,350 rows as the baseline
pre = cp.build_preprocessor(scale=False)       # trees; scale=True for linear / kNN / SVM
```

- Baseline to beat (validation, class 1): Precision 0.458, Recall 0.647, F1 0.536 (logistic regression, balanced).
- Without the two anomaly flags F1 drops to 0.472 — report both variants (see notebook section 6).
- Error analysis can start from `artifacts/participant2/baseline_validation_predictions.csv`.
- Use `cp.load_test_labels()` only once, for the final evaluation of the chosen model.
