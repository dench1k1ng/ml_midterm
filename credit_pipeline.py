"""Shared data loading, split and preprocessing for the Credit Default project.

Participant 2 owns this module. Participant 3 imports it so that every model is
trained on the same rows, validated on the same rows and sees identical
preprocessing. Every fitted parameter (medians, scaler statistics, category
levels) is learned inside ``Pipeline.fit`` on the training portion only.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = 'Credit Default'
RANDOM_STATE = 42
VALID_SIZE = 0.2
ARCHIVE = 'Project_eng.zip'
SPLIT_FILE = Path('artifacts') / 'participant2' / 'split_indices.csv'

SCORE_SCALE_LIMIT = 850          # EDA section 5: 357 train scores in 6060..7510, all divisible by 10
LOAN_SENTINEL = 99_999_999       # EDA section 5: repeated exact value, 783 train rows

TENURE_ORDER = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
    '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10,
}

# Columns produced by DomainCleaner, grouped by how the ColumnTransformer treats them.
MONEY_COLS = ['Annual Income', 'Maximum Open Credit', 'Current Loan Amount',
              'Current Credit Balance', 'Monthly Debt']
OTHER_NUMERIC_COLS = ['Tax Liens', 'Number of Open Accounts', 'Years of Credit History',
                      'Number of Credit Problems', 'Months since last delinquent',
                      'Bankruptcies', 'Credit Score', 'Years in current job']
FLAG_COLS = ['income_score_missing', 'delinquent_missing', 'tenure_missing',
             'score_scale_flag', 'loan_sentinel_flag']
CATEGORICAL_COLS = ['Home Ownership', 'Purpose', 'Term']
ANOMALY_FLAGS = ['score_scale_flag', 'loan_sentinel_flag']


def load_data(archive_path: str | Path = ARCHIVE):
    """Return (X_train_full, y_train_full, X_test). Test labels are never returned."""
    with ZipFile(archive_path) as archive:
        train = pd.read_csv(archive.open('project-5/train_data.csv'))
        test = pd.read_csv(archive.open('project-5/test_data.csv'),
                           usecols=lambda column: column != TARGET)
    X_train_full = train.drop(columns=[TARGET])
    y_train_full = train[TARGET]
    X_test = test  # target never materialised; all 750 rows kept, original order
    return X_train_full, y_train_full, X_test


def load_test_labels(archive_path: str | Path = ARCHIVE) -> pd.Series:
    """Final evaluation only (participant 3), after every model decision is frozen."""
    with ZipFile(archive_path) as archive:
        return pd.read_csv(archive.open('project-5/test_data.csv'))[TARGET]


def make_split(y: pd.Series):
    """Stratified 80/20 train/validation split of the original training file."""
    idx_tr, idx_va = train_test_split(y.index, test_size=VALID_SIZE, stratify=y,
                                      random_state=RANDOM_STATE)
    return idx_tr.sort_values(), idx_va.sort_values()


def load_split(path: str | Path = SPLIT_FILE):
    """Read the saved split so other notebooks reuse exactly the same rows."""
    split = pd.read_csv(path)
    idx_tr = pd.Index(split.loc[split['part'].eq('train'), 'row'])
    idx_va = pd.Index(split.loc[split['part'].eq('valid'), 'row'])
    return idx_tr, idx_va


class DomainCleaner(BaseEstimator, TransformerMixin):
    """Stateless, row-wise fixes motivated by the EDA. Learns nothing from data.

    * Credit Score > 850 -> divide by 10, missing, or raw (explicit hypothesis).
    * Current Loan Amount == 99999999 -> NaN (treated as unknown), flag kept.
    * Years in current job -> ordinal 0..10.
    * One indicator for income/score missingness (their masks are identical in EDA),
      plus indicators for delinquency and tenure missingness.
    """

    def __init__(self, use_anomaly_flags: bool = True, score_policy: str = "divide"):
        self.use_anomaly_flags = use_anomaly_flags
        self.score_policy = score_policy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.score_policy not in {'divide', 'missing', 'raw'}:
            raise ValueError('score_policy must be divide, missing or raw')
        X = X.copy()
        score_high = X['Credit Score'].gt(SCORE_SCALE_LIMIT)
        loan_sentinel = X['Current Loan Amount'].eq(LOAN_SENTINEL)
        X['income_score_missing'] = X['Annual Income'].isna().astype(int)
        X['delinquent_missing'] = X['Months since last delinquent'].isna().astype(int)
        X['tenure_missing'] = X['Years in current job'].map(TENURE_ORDER).isna().astype(int)
        X['score_scale_flag'] = score_high.astype(int)
        X['loan_sentinel_flag'] = loan_sentinel.astype(int)
        if self.score_policy == 'divide':
            X.loc[score_high, 'Credit Score'] = X.loc[score_high, 'Credit Score'] / 10
        elif self.score_policy == 'missing':
            X.loc[score_high, 'Credit Score'] = np.nan
        X.loc[loan_sentinel, 'Current Loan Amount'] = np.nan
        X['Years in current job'] = X['Years in current job'].map(TENURE_ORDER).astype(float)
        if not self.use_anomaly_flags:
            X = X.drop(columns=ANOMALY_FLAGS)
        return X


def build_preprocessor(use_anomaly_flags: bool = True, scale: bool = True,
                       min_frequency: int = 20, score_policy: str = "divide") -> Pipeline:
    """Unfitted preprocessing: DomainCleaner followed by a ColumnTransformer.

    ``scale=False`` skips log/StandardScaler for tree models (participant 3).
    ``min_frequency`` is learned per training fold; ``score_policy`` exposes the
    unconfirmed mixed-scale hypothesis for training-only sensitivity checks.
    """
    flags = FLAG_COLS if use_anomaly_flags else [c for c in FLAG_COLS if c not in ANOMALY_FLAGS]
    if scale:
        money = make_pipeline(SimpleImputer(strategy='median'),
                              FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
                              StandardScaler())
        numeric = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    else:
        money = SimpleImputer(strategy='median')
        numeric = SimpleImputer(strategy='median')
    categorical = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=min_frequency, sparse_output=False),
    )
    columns = ColumnTransformer([
        ('money', money, MONEY_COLS),
        ('numeric', numeric, OTHER_NUMERIC_COLS),
        ('flags', 'passthrough', flags),
        ('categorical', categorical, CATEGORICAL_COLS),
    ], verbose_feature_names_out=False)
    return Pipeline([('clean', DomainCleaner(use_anomaly_flags, score_policy)), ('columns', columns)])


# Part 3: candidate models, class-1 metrics and training-only threshold tuning.
def make_models(use_anomaly_flags: bool = True) -> dict[str, Pipeline]:
    """Unfitted pipelines; trees skip log/scaling, logistic regression keeps it."""
    def pipe(model, scale):
        return Pipeline([('pre', build_preprocessor(use_anomaly_flags, scale=scale)),
                         ('model', model)])
    return {
        'logreg': pipe(LogisticRegression(class_weight='balanced', max_iter=1000), True),
        'random_forest': pipe(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight='balanced_subsample',
            n_jobs=-1, random_state=RANDOM_STATE), False),
        'hist_gb': pipe(HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight='balanced',
            random_state=RANDOM_STATE), False),
    }


def class1_metrics(y_true, y_pred) -> dict[str, float]:
    kw = dict(pos_label=1, zero_division=0)
    return {'precision': float(precision_score(y_true, y_pred, **kw)),
            'recall': float(recall_score(y_true, y_pred, **kw)),
            'f1': float(f1_score(y_true, y_pred, **kw))}


def tune_threshold(pipe: Pipeline, X, y) -> TunedThresholdClassifierCV:
    """Pick the F1-optimal cut-off on 5 folds of (X, y) only, then refit on all of it."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    return TunedThresholdClassifierCV(pipe, scoring='f1', cv=cv,
                                      store_cv_results=True).fit(X, y)
