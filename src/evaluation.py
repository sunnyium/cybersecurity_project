"""
Evaluation utilities for the CIC-IDS2017 multi-class problem (7 LabelGroups).

Headline metrics:
    summary_metrics() | accuracy alongside macro precision/recall/F1
    per_class_report() | precision/recall/F1/support for every class
    constant_baseline() | the "flag nothing" reference, computed not fitted
    class_names() | decoded class labels in encoder order

Baseline ladder:
    evaluate_model() | fit one model, score it on the test set
    baseline_ladder() | majority -> logistic -> XGBoost, scored identically
    plot_baseline_ladder() | accuracy against macro-F1 for each rung

Error structure:
    plot_confusion_matrix() | row-normalised, so each row reads as recall
    top_confusions() | the largest off-diagonal cells, ranked

Feature attribution:
    feature_importance() | gain-based importance as a sorted table
    plot_feature_importance() | the top-N features

Stability:
    cross_val_macro_f1() | stratified k-fold macro-F1, mean and spread

Accuracy, macro-F1 and the per-class results are reported. Keep in mind
that BENIGN makes up 83% of data. 

Every model is scored through models.predict_in_chunks: the test split keeps
the true class distribution and is never subsampled.
"""

from __future__ import annotations

import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

import models as md
import plotting as plot
import split_dataset as sd

RANDOM_STATE = sd.RANDOM_STATE

# Rungs in the order they should be reported, weakest first
LADDER = ("majority", "logistic", "xgboost")

BUILDERS = {
    "majority": md.build_majority_baseline,
    "logistic": md.build_logistic_baseline,
    "xgboost": md.build_xgboost,
}

# Display names for the ladder table and figures. 
# Majority is the class it predicts.
LADDER_LABELS = {
    "majority": "Majority",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
}

CV_SPLITS = 5

# White -> PRIMARY
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "ids_blues", ["#ffffff", plot.PRIMARY]
)


# Helper functions
def class_names(split: dict) -> list[str]:
    """Class labels in label-encoder order, so index i is class i."""
    return [str(c) for c in split["label_encoder"].classes_]


def _class_index(names: Sequence[str]) -> list[int]:
    return list(range(len(names)))


# 1. Headline metrics
def summary_metrics(y_true, y_pred) -> dict:
    """Accuracy plus the macro averages.
    Macro averaging gives each class equal weight so the six attack families
    are not overweighted by BENIGN."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MacroPrecision": float(p),
        "MacroRecall": float(r),
        "MacroF1": float(f),
    }


def per_class_report(y_true, y_pred, names: Sequence[str],
    decimals: int = 3,) -> pd.DataFrame:
    """Per-class precision/recall/F1/support.
    A class can be missed (low recall) or over-flagged (low precision)."""
    rep = classification_report(
        y_true, y_pred, labels=_class_index(names), target_names=list(names),
        output_dict=True, zero_division=0,
    )
    rows = [
        {
            "Class": n,
            "Precision": round(rep[n]["precision"], decimals),
            "Recall": round(rep[n]["recall"], decimals),
            "F1": round(rep[n]["f1-score"], decimals),
            "Support": int(rep[n]["support"]),
        }
        for n in names
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("Support", ascending=False)
        .reset_index(drop=True)
    )


def constant_baseline(y_true, names: Sequence[str],
    cls: str | int | None = None) -> dict:
    """Metrics for predicting a single class for every row, computed without fitting.
    This baseline reads the class directly from `y_true`, so a limited training set
    cannot influence the result. The `cls` argument defaults to the most common
    class in `y_true` and may be given as a class label or an encoded index."""

    y_true = np.asarray(y_true)
    if cls is None:
        idx = int(np.argmax(np.bincount(y_true, minlength=len(names))))
    elif isinstance(cls, str):
        idx = list(names).index(cls)
    else:
        idx = int(cls)

    pred = np.full(len(y_true), idx, dtype=y_true.dtype)
    out = {"Model": f"Always {names[idx]}"}
    out.update(summary_metrics(y_true, pred))
    return out


# 2. Baseline ladder
def evaluate_model(model, split: dict, name: str = "model",
    names: Sequence[str] | None = None, verbose: bool = True,
    ) -> tuple[dict, np.ndarray]:
    """Fit `model` on the training fold and score it on the test fold.
    Returns (metrics, predictions). Fit time is also recorded as a point of
    comparison, specifically between the logistic and XGBoost rungs. """

    t0 = time.time()
    model.fit(split["X_train"], split["y_train"])
    fit_s = time.time() - t0

    pred = md.predict_in_chunks(model, split["X_test"])

    label = LADDER_LABELS.get(name, name)
    if names is not None and len(np.unique(pred)) == 1:
        label = f"{label} ({names[int(pred[0])]})"

    metrics = {"Model": label}
    metrics.update(summary_metrics(split["y_test"], pred))
    metrics["FitSeconds"] = round(fit_s, 1)

    if verbose:
        print(f"{metrics['Model']:<20} macro-F1 {metrics['MacroF1']:.4f}   "
              f"accuracy {metrics['Accuracy']:.4f}   ({fit_s:.0f}s)")
    return metrics, pred


def baseline_ladder(split: dict, names: Sequence[str] = LADDER,
    decimals: int = 4, verbose: bool = True,) -> dict:
    """Fit each rung on the same split and score it in the same way.
    Returns a dict with "table", "models", and "predictions".

    LADDER is an ordered set of rung names. The majority rung is the 
    control which learns and predicts a single class. 

    Note: DummyClassifier chooses the most frequent class which may
    not be BENIGN if the training set is capped below the number of 
    DoS attacks. 

    The logistic rung is slow on an uncapped training set. Pass a 
    capped split or drop "logistic" from `names` when iterating."""

    rows, fitted, preds = [], {}, {}
    labels = class_names(split)

    for name in names:
        model = BUILDERS[name]()
        metrics, pred = evaluate_model(model, split, name=name, names=labels,
                                       verbose=verbose)
        rows.append(metrics)
        fitted[name] = model
        preds[name] = pred

    table = pd.DataFrame(rows)[
        ["Model", "Accuracy", "MacroPrecision", "MacroRecall", "MacroF1", "FitSeconds"]
    ]
    numeric = ["Accuracy", "MacroPrecision", "MacroRecall", "MacroF1"]
    table[numeric] = table[numeric].round(decimals)

    return {"table": table, "models": fitted, "predictions": preds}


def plot_baseline_ladder(table: pd.DataFrame,
    save_name: str = "baseline_ladder.png",) -> str:
    """Accuracy against macro-F1 for every rung, on one axis."""
    rows = table.iloc[::-1]                    # weakest rung at the bottom
    y = np.arange(len(rows))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(rows) + 2.4))
    ax.barh(y + height / 2, rows["Accuracy"], height=height,
            color=plot.MUTED, label="Accuracy", zorder=3)
    ax.barh(y - height / 2, rows["MacroF1"], height=height,
            color=plot.PRIMARY, label="Macro-F1", zorder=3)

    for yi, (acc, f1) in enumerate(zip(rows["Accuracy"], rows["MacroF1"])):
        ax.annotate(f"{acc:.3f}", xy=(acc, yi + height / 2), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=plot.TICK_SIZE, color="0.25")
        ax.annotate(f"{f1:.3f}", xy=(f1, yi - height / 2), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=plot.TICK_SIZE, color="0.25")

    ax.set_xlim(0, 1.12)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["Model"])
    plot.style_axes(ax, grid_axis="x")

    ax.set_xlabel("score", fontsize=plot.LABEL_SIZE, labelpad=8)
    ax.legend(frameon=False, fontsize=plot.TICK_SIZE, loc="lower right", ncols=2)

    worst = table.loc[table["MacroF1"].idxmin()]
    best = table.loc[table["MacroF1"].idxmax()]
    plot.add_titles(
        ax,
        "Baseline ladder",
        f"{worst['Model']} already scores {worst['Accuracy']:.2f} accuracy at "
        f"{worst['MacroF1']:.2f} macro-F1 - {best['Model']} reaches "
        f"{best['MacroF1']:.2f}",
    )
    return plot.save_figure(fig, save_name)


# 3. Error structure
def plot_confusion_matrix(y_true, y_pred, names: Sequence[str],
    normalize: bool = True, save_name: str = "confusion_matrix.png",) -> str:
    """Confusion matrix, row-normalised by default.

    Row-normalising makes the diagonal read directly as per-class 
    recall and rare classes stay legible next to BENIGN."""
    cm = confusion_matrix(y_true, y_pred, labels=_class_index(names))

    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        disp = np.nan_to_num(disp)
        fmt = lambda v: f"{v:.2f}"
        vmax = 1.0
    else:
        disp = cm.astype(float)
        fmt = lambda v: f"{int(v):,}"
        vmax = disp.max()

    n = len(names)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2.6, 0.95 * n + 2.4))
    im = ax.imshow(disp, cmap=CONFUSION_CMAP, vmin=0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=plot.TICK_SIZE)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=plot.TICK_SIZE)
    ax.set_xlabel("predicted", fontsize=plot.LABEL_SIZE, labelpad=8)
    ax.set_ylabel("true", fontsize=plot.LABEL_SIZE, labelpad=8)

    plot.style_axes(ax, grid_axis=None, hide=("top", "right", "left", "bottom"),
                    flat_ticks="both")

    for i in range(n):
        for j in range(n):
            if disp[i, j] == 0:
                continue
            ax.text(j, i, fmt(disp[i, j]), ha="center", va="center",
                    fontsize=plot.TICK_SIZE - 1,
                    color="white" if disp[i, j] > 0.55 * vmax else "0.25")

    recall = np.diag(disp) if normalize else np.diag(disp) / cm.sum(axis=1)
    weakest = names[int(np.argmin(recall))]
    subtitle = (
        f"row-normalised - the diagonal is per-class recall, weakest is "
        f"{weakest} at {recall.min():.2f}"
        if normalize else
        f"raw counts - the diagonal is correct predictions, weakest recall is "
        f"{weakest} at {recall.min():.2f}"
    )
    plot.add_titles(ax, "Confusion matrix", subtitle)
    return plot.save_figure(fig, save_name)


def top_confusions(y_true, y_pred, names: Sequence[str],
    top: int = 5,) -> pd.DataFrame:
    """The largest off-diagonal cells, ranked by share of the true class."""
    cm = confusion_matrix(y_true, y_pred, labels=_class_index(names))
    support = cm.sum(axis=1, keepdims=True)

    rows = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or cm[i, j] == 0:
                continue
            rows.append(
                {
                    "TrueClass": names[i],
                    "PredictedClass": names[j],
                    "Count": int(cm[i, j]),
                    "ShareOfTrueClass": round(
                        float(cm[i, j] / support[i, 0]) if support[i, 0] else 0.0, 4
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("ShareOfTrueClass", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


# 4. Feature attribution
def feature_importance(model, feature_names: Sequence[str],
    top: int | None = None,) -> pd.DataFrame:
    """Gain-based importance as a sorted table.

    Tree importance says which features the model leaned on, not which
    features cause an attack."""
    imp = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "Importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    return imp.head(top) if top else imp


def plot_feature_importance(imp: pd.DataFrame, top: int = 15,
    save_name: str = "feature_importance.png",) -> str:
    """Horizontal bars for the top-N features by gain."""
    rows = imp.head(top).iloc[::-1]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 2.2))
    ax.barh(y, rows["Importance"], color=plot.PRIMARY, height=0.72, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["Feature"], fontsize=plot.TICK_SIZE)
    plot.style_axes(ax, grid_axis="x")
    ax.set_xlabel("importance (gain)", fontsize=plot.LABEL_SIZE, labelpad=8)

    share = float(imp.head(top)["Importance"].sum())
    plot.add_titles(
        ax,
        f"Top {len(rows)} features by gain",
        f"these {len(rows)} of {len(imp)} features carry {share:.0%} of total gain",
    )
    return plot.save_figure(fig, save_name)


# 5. Stability
def cross_val_macro_f1(X, y, build=md.build_xgboost, n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE, verbose: bool = True,) -> dict:
    """Stratified k-fold macro-F1, to remove the chance a single split is lucky.
    Returns a dict with "scores", "mean" and "std".
    
    A tight spread across folds would prove a single-split score is real."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = build()
        model.fit(X[tr], y[tr])
        pred = md.predict_in_chunks(model, X[te])
        score = f1_score(y[te], pred, average="macro", zero_division=0)
        scores.append(float(score))
        if verbose:
            print(f"fold {k}/{n_splits}  macro-F1 {score:.4f}")

    scores = np.asarray(scores, dtype=float)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


if __name__ == "__main__":
    split = sd.load_split()
    names = class_names(split)
    print(f"train {len(split['y_train']):,} rows   test {len(split['y_test']):,} rows")
    print(f"classes: {names}\n")

    ladder = baseline_ladder(split, names=("majority", "xgboost"))
    print(f"\n{ladder['table'].to_string(index=False)}\n")

    pred = ladder["predictions"]["xgboost"]
    print(per_class_report(split["y_test"], pred, names).to_string(index=False))

    print(f"\n{plot_baseline_ladder(ladder['table'])}")
    print(plot_confusion_matrix(split["y_test"], pred, names))

    imp = feature_importance(ladder["models"]["xgboost"], split["feature_names"])
    print(plot_feature_importance(imp))
