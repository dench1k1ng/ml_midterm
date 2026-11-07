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
from sklearn.model_selection import train_test_split
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
        test = pd.read_csv(archive.open('project-5/test_data.csv'))
    X_train_full = train.drop(columns=[TARGET])
    y_train_full = train[TARGET]
    X_test = test.drop(columns=[TARGET])  # all 750 rows kept, original order
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

    * Credit Score > 850 -> divided by 10 (hypothesised x10 entry), flag kept.
    * Current Loan Amount == 99999999 -> NaN (treated as unknown), flag kept.
    * Years in current job -> ordinal 0..10.
    * One indicator for income/score missingness (their masks are identical in EDA),
      plus indicators for delinquency and tenure missingness.
    """

    def __init__(self, use_anomaly_flags: bool = True):
        self.use_anomaly_flags = use_anomaly_flags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        score_high = X['Credit Score'].gt(SCORE_SCALE_LIMIT)
        loan_sentinel = X['Current Loan Amount'].eq(LOAN_SENTINEL)
        X['income_score_missing'] = X['Annual Income'].isna().astype(int)
        X['delinquent_missing'] = X['Months since last delinquent'].isna().astype(int)
        X['tenure_missing'] = X['Years in current job'].isna().astype(int)
        X['score_scale_flag'] = score_high.astype(int)
        X['loan_sentinel_flag'] = loan_sentinel.astype(int)
        X.loc[score_high, 'Credit Score'] = X.loc[score_high, 'Credit Score'] / 10
        X.loc[loan_sentinel, 'Current Loan Amount'] = np.nan
        X['Years in current job'] = X['Years in current job'].map(TENURE_ORDER).astype(float)
        if not self.use_anomaly_flags:
            X = X.drop(columns=ANOMALY_FLAGS)
        return X


def build_preprocessor(use_anomaly_flags: bool = True, scale: bool = True) -> Pipeline:
    """Unfitted preprocessing: DomainCleaner followed by a ColumnTransformer.

    ``scale=False`` skips log/StandardScaler for tree models (participant 3).
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
        OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, sparse_output=False),
    )
    columns = ColumnTransformer([
        ('money', money, MONEY_COLS),
        ('numeric', numeric, OTHER_NUMERIC_COLS),
        ('flags', 'passthrough', flags),
        ('categorical', categorical, CATEGORICAL_COLS),
    ], verbose_feature_names_out=False)
    return Pipeline([('clean', DomainCleaner(use_anomaly_flags)), ('columns', columns)])
