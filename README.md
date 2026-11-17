# Final Project — SMS Spam Detection

Binary classification of SMS messages into `ham` and `spam` on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 English messages, CC BY 4.0). The team submits one notebook: `Final_Denis_Shin_Abylay_Latiyev_Nikita_Vassilenko.ipynb`.

| Part | Scope | Status |
|---|---|---|
| 1 — Data & classical ML | deduplication, shared split, EDA, meta-features, NB / LogReg / RF / LinearSVC / HistGradientBoosting | done |
| 2 — Deep learning | neural models / transformers, weighted loss, threshold tuning | to do |
| 3 — Evaluation & demo | team benchmark, error analysis, demo | to do |

Main metric: F1 for `spam`, with Precision and PR-AUC reported alongside.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock.txt
jupyter nbconvert --to notebook --execute --inplace Final_Denis_Shin_Abylay_Latiyev_Nikita_Vassilenko.ipynb
```

Run from the repository root. The notebook downloads NLTK stop-words on first run if they are missing.

## Files for Parts 2–3

- `data/SMSSpamCollection` — original UCI file.
- `data/sms_dedup.csv` — `idx, label, text` after removing 415 duplicates, 5,159 messages left. Duplicates are matched after lower-casing and collapsing whitespace.
- `artifacts/split.csv` — `idx, split`: stratified train / val / test = 3,611 / 516 / 1,032, seed 42. Use `val` for early stopping and threshold tuning, `test` only for the final evaluation.
- `artifacts/participant1/test_predictions.csv` — `idx, y_true, score_nb, score_logreg, score_rf, score_svc, score_hgb` on the test split (`score_svc` is a decision score, the others are spam probabilities).
- `artifacts/participant1/cv_results.csv` — best hyper-parameters and CV F1 per model.
- `artifacts/participant1/top_features.csv` — top-20 features per model for error analysis.

```python
import pandas as pd
df = pd.read_csv("data/sms_dedup.csv").merge(pd.read_csv("artifacts/split.csv"), on="idx")
train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
```

## Part 1 results (test, spam class)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| LinearSVC | 0.992 | 0.953 | 0.972 | 0.992 |
| LogReg | 0.992 | 0.945 | 0.968 | 0.994 |
| Random Forest | 0.984 | 0.938 | 0.960 | 0.991 |
| HistGradientBoosting | 1.000 | 0.922 | 0.959 | 0.990 |
| Multinomial NB | 0.959 | 0.914 | 0.936 | 0.978 |
