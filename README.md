# ml_endterm — Project 5: Loan Approval (`Credit Default`)

Data: `Project_eng.zip` (`project-5/train_data.csv` — 6,750 rows, `project-5/test_data.csv` — 750 rows).
Test rows must not be deleted.

- Progress and next steps: [PROGRESS.md](PROGRESS.md)
- Guide for participant 2: [2_Abylay_Latiyev_Guide.md](2_Abylay_Latiyev_Guide.md)

| Part | Owner | Deliverable | Status |
|---|---|---|---|
| Data audit & EDA | Denis Shin | `1_Denis_Shin_EDA.ipynb` | done |
| Preprocessing, split, baseline | Abylay Latiyev | `credit_pipeline.py`, `2_Abylay_Latiyev_Preprocessing_Baseline.ipynb`, `artifacts/participant2/` | done |
| Extra models, comparison, error analysis | Nikita Vassilenko | `participant3.py`, `3_Nikita_Vassilenko_Models_Comparison.ipynb`, `artifacts/participant3/` | done |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python run_notebook.py
python run_notebook.py 3_Nikita_Vassilenko_Models_Comparison.ipynb
python -m unittest discover -s tests -v
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
- Fixed baseline without the flags: F1 0.472. Removing flags alone does not establish absence of leakage.
- Training-only search: 120 configurations × 5 folds; chosen C=0.1, balanced, rare count=50, high-score policy=missing. Validation F1 0.533 (no-flags ablation: 0.463), so tuning did not improve on the fixed baseline. Full search and tuned predictions are in `artifacts/participant2/`; see the guide for reconstruction.
- Denis explored the full original training target before the split: validation is not fully blind. Anomaly provenance remains unconfirmed.
- Error analysis can start from `artifacts/participant2/baseline_validation_predictions.csv`.
- Use `cp.load_test_labels()` only once, for the final evaluation of the chosen model.

## Result

Chosen model: logistic regression with anomaly flags. Test (class 1): Precision 0.399, Recall 0.633, F1 0.490, ROC AUC 0.756. Tree ensembles win only without the flags (validation F1 0.499 vs 0.478). Details in the part 3 notebook.
