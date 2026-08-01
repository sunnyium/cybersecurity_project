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
    This model is used as a baseline and a trained model must beat it
    even though its accuracy is high and macro-F1 is near zero"""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_baseline(max_iter: int = 200,
    random_state: int = RANDOM_STATE,) -> Pipeline:
    """Scaled multinomial logistic regression (linear reference point)
    StandardScaler is fit inside each training fold and never sees test rows."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])


def build_xgboost(random_state: int = RANDOM_STATE, **overrides) -> XGBClassifier:
    """Histogram-based gradient boosting for the 7-class problem.
    tree_method='hist' with max_bin=128 sorts each feature inot bins instead
    of sorting every split candidate. 
    This makes ~2M rows trainable on a standard computer."""
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
    The test set is never subsampled and stays large. Predicting in 
    slices bounds peak memory to chunks of scores."""
    parts = [model.predict(X[s: s + chunk]) for s in range(0, len(X), chunk)]
    return np.concatenate(parts)


def predict_proba_in_chunks(model, X: np.ndarray,
    chunk: int = PREDICT_CHUNK,) -> np.ndarray:
    """Class probabilities for X as an (n_rows, n_classes) array, chunked.
    The output is n_rows x 7 float64 using less memory than a label vector."""
    parts = [model.predict_proba(X[s: s + chunk]) for s in range(0, len(X), chunk)]
    return np.concatenate(parts)
