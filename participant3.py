"""Participant 3: candidate models, class-1 metrics and training-only threshold tuning."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

import credit_pipeline as cp


def make_models(use_anomaly_flags: bool = True) -> dict[str, Pipeline]:
    """Unfitted pipelines; trees skip log/scaling, logistic regression keeps it."""
    def pipe(model, scale):
        return Pipeline([('pre', cp.build_preprocessor(use_anomaly_flags, scale=scale)),
                         ('model', model)])
    return {
        'logreg': pipe(LogisticRegression(class_weight='balanced', max_iter=1000), True),
        'random_forest': pipe(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight='balanced_subsample',
            n_jobs=-1, random_state=cp.RANDOM_STATE), False),
        'hist_gb': pipe(HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight='balanced',
            random_state=cp.RANDOM_STATE), False),
    }


def class1_metrics(y_true, y_pred) -> dict[str, float]:
    kw = dict(pos_label=1, zero_division=0)
    return {'precision': float(precision_score(y_true, y_pred, **kw)),
            'recall': float(recall_score(y_true, y_pred, **kw)),
            'f1': float(f1_score(y_true, y_pred, **kw))}


def tune_threshold(pipe: Pipeline, X, y) -> TunedThresholdClassifierCV:
    """Pick the F1-optimal cut-off on 5 folds of (X, y) only, then refit on all of it."""
    cv = StratifiedKFold(5, shuffle=True, random_state=cp.RANDOM_STATE)
    return TunedThresholdClassifierCV(pipe, scoring='f1', cv=cv,
                                      store_cv_results=True).fit(X, y)
