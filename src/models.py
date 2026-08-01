"""
Model definitions for the CIC-IDS2017 multi-class problem (7 LabelGroups).

Baseline ladder:
    build_majority_baseline() | always predicts BENIGN, the score any real 
        model must beat
    build_logistic_baseline() | scaled multinomial logistic regression
    build_xgboost() | primary model training function with a histogram
        based gradient boosting

Prediction:
    predict_in_chunks() | labels for a large X without a single huge call
    predict_proba_in_chunks() | same as predict_in_chunks with chunked
        probabilty predictions for the (n_rows, 7) matrix

Scaling: the flow features span wildly different magnitudes the logistic 
    baseline uses StandardScaler fit on train data only because XGBoost 
    is scale variant.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import split_dataset as sd

RANDOM_STATE = sd.RANDOM_STATE

# Rows per chunk in the predict_* helpers. 250k x 7 float64 probabilities is
# ~14MB per chunk, which keeps prediction on the uncapped test set bounded.
PREDICT_CHUNK = 250_000


# 1. Baseline ladder
def build_majority_baseline() -> DummyClassifier:
    """Predict the most frequent class (BENIGN) for everything.

    Not a real model -- it is the control. Its accuracy is high and its
    macro-F1 is near zero, which is the clearest possible demonstration that
    accuracy is the wrong headline metric on this dataset.
    """
    return DummyClassifier(strategy="most_frequent")


def build_logistic_baseline(max_iter: int = 200,
    random_state: int = RANDOM_STATE,) -> Pipeline:
    """Scaled multinomial logistic regression -- the linear reference point.

    Wrapped in a Pipeline so the scaler is fit inside each training fold and
    never sees test rows.

    Note on cost: lbfgs on the uncapped training set (~2M rows x 69 features)
    is by far the slowest rung of the ladder. Fit it on a capped training set
    (see split_dataset.DEFAULT_TRAIN_CAP) unless there is a reason not to.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])


def build_xgboost(random_state: int = RANDOM_STATE, **overrides) -> XGBClassifier:
    """Histogram-based gradient boosting for the 7-class problem.

    tree_method='hist' with max_bin=128 buckets each feature instead of
    sorting every split candidate, which is what makes ~2M rows tractable on
    a modest machine. num_class is inferred from y at fit time.

    Any parameter can be overridden by keyword, e.g. build_xgboost(max_depth=6).
    """
    params = dict(
        n_estimators=120,
        max_depth=8,
        learning_rate=0.3,
        tree_method="hist",
        max_bin=128,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=2,
        random_state=random_state,
    )
    params.update(overrides)
    return XGBClassifier(**params)


# 2. Memory-bounded prediction
def predict_in_chunks(model, X: np.ndarray, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """Predicted labels for X, one chunk at a time.

    The test set is never subsampled (split_dataset keeps its true class
    distribution), so it stays large; predicting in slices bounds peak memory
    to one chunk of intermediate scores rather than the whole matrix.
    """
    parts = [model.predict(X[s: s + chunk]) for s in range(0, len(X), chunk)]
    return np.concatenate(parts)


def predict_proba_in_chunks(model, X: np.ndarray,
    chunk: int = PREDICT_CHUNK,) -> np.ndarray:
    """Class probabilities for X as an (n_rows, n_classes) array, chunked.

    Matters more than predict_in_chunks: the output itself is n_rows x 7
    float64, roughly 8x the memory of a label vector.
    """
    parts = [model.predict_proba(X[s: s + chunk]) for s in range(0, len(X), chunk)]
    return np.concatenate(parts)
