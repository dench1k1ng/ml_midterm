# Final Project — SMS Spam Detection

Binary classification of SMS messages into `ham` and `spam` on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 English messages, CC BY 4.0). The team submits one notebook, `Final_Denis_Shin_Abylay_Latiyev_Nikita_Vassilenko.ipynb`, and the [final presentation](SMS_Spam_Detection_Final.pptx).

| Part | Scope | Status |
|---|---|---|
| 1 — Data & classical ML | deduplication, shared split, EDA, meta-features, NB / LogReg / RF / LinearSVC / HistGradientBoosting | done |
| 2 — Deep learning | BPE 1D-CNN, WordPiece BERT-Tiny, weighted loss, validation threshold tuning | done |
| 3 — Evaluation & demo | shared benchmark, feature and error analysis, Streamlit demo, presentation | done |

Main metric: F1 for `spam`, with Precision and PR-AUC reported alongside.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Final_Denis_Shin_Abylay_Latiyev_Nikita_Vassilenko.ipynb
```

Run from the repository root. The commands install the CPU build of PyTorch. Part 2 also uses CUDA automatically if a compatible PyTorch build and GPU are available. On first run the notebook downloads NLTK stop-words and the pretrained BERT-Tiny checkpoint if they are missing. To rerun just Part 2, start at section 2.1. It reads the checked-in dataset and split.

## Run the Streamlit demo

After executing the full notebook, install the demo dependency in the same virtual environment and start the app:

```bash
python -m pip install -r requirements-demo.txt
streamlit run demo.py
```

Section 3.4 exports the fine-tuned BERT-Tiny checkpoint locally. To recreate it without rerunning the classical models, run Part 2 and then Part 3 in the same kernel. The app applies the validation-tuned threshold and shows the ham/spam decision and uncalibrated spam probability.

## Files for Parts 2–3

- `data/SMSSpamCollection` — original UCI file.
- `data/sms_dedup.csv` — `idx, label, text` after removing 415 duplicates, 5,159 messages left. Duplicates are matched after lower-casing and collapsing whitespace.
- `artifacts/split.csv` — `idx, split`: stratified train / val / test = 3,611 / 516 / 1,032, seed 42. Use `val` for early stopping and threshold tuning, `test` only for the final evaluation.
- `artifacts/participant1/test_predictions.csv` — `idx, y_true, score_nb, score_logreg, score_rf, score_svc, score_hgb` on the test split (`score_svc` is a decision score, the others are spam probabilities).
- `artifacts/participant1/cv_results.csv` — best hyper-parameters and CV F1 per model.
- `artifacts/participant1/top_features.csv` — top-20 features per model for error analysis.
- `artifacts/participant2/test_predictions.csv` — `idx, y_true, score_cnn, pred_cnn, score_bert_tiny, pred_bert_tiny`. Join on `idx` for the team benchmark. The `pred_*` columns use each model's validation-tuned threshold.
- `artifacts/participant2/test_metrics.csv` — test Precision, Recall, F1 and PR-AUC (average precision) for spam, with the selected epoch, threshold and validation F1.
- `artifacts/participant3/benchmark.csv` — shared test metrics and FP/FN counts for seven models and two dummy baselines.
- `artifacts/participant3/error_examples.csv` — six annotated test errors from LinearSVC and BERT-Tiny.

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
| Dummy (random by class shares) | 0.212 | 0.227 | 0.219 | 0.144 |
| Dummy (always ham) | 0.000 | 0.000 | 0.000 | 0.124 |

## Sources

- Almeida, T. A., Gómez Hidalgo, J. M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. DocEng 2011. Dataset: [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), CC BY 4.0.
- Gómez Hidalgo, J. M., Almeida, T. A., Yamakami, A. (2012). On the Validity of a New SMS Spam Collection. ICMLA 2012. Cited for the duplicate analysis.
- Libraries: scikit-learn, NLTK (stop-words, Porter stemmer), pandas, matplotlib.
- Part 2: PyTorch, Hugging Face Transformers and Tokenizers, [Google BERT-Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) (Turc et al., 2019).

## AI assistance

Part 1 was written with Claude Code (Anthropic), which helped with code, notebook text and review. Every team member checked the results and can explain the submitted work.
Part 2 was implemented and run with Codex (OpenAI).
Part 3 was implemented and run with Codex (OpenAI).
