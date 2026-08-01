"""
Fit and persist the multi-class XGBoost model for CIC-IDS2017.

    train() | fit XGBoost on the training fold and write it to data/
    load_model() | read a persisted model back
    load_metadata() | class order, feature order and provenance of that model
    main() | command line entry point

Run after build_dataset.py has written merge_complete.parquet:

    python src/train.py                # BENIGN capped to TRAIN_CAP
    python src/train.py --cap 0        # uncapped training fold
    python src/train.py --no-score     # skip the post-fit sanity check

The model is persisted because notebook 3 needs it for the per-class report,
the confusion matrix and feature importance. Refitting on every kernel restart
costs minutes and buys nothing.

The saved model stores class indices, not class names, so a sidecar json
records the encoder order alongside the feature order and the cap that was
used. Reading importances or predictions back without that order is how a
silent class-mix-up happens.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from xgboost import XGBClassifier

import build_dataset as bd
import evaluation as ev
import models as md
import split_dataset as sd

MODEL_NAME = "xgb_multiclass.json"
META_NAME = "xgb_multiclass_meta.json"

# BENIGN rows kept in the training fold; None disables the cap.
#
# 200k is deliberately above the DoS count in the training fold (~155k).
# subsample_majority_train caps whichever class is largest, so capping below
# the DoS count would hand the majority to DoS: the majority rung of the
# baseline ladder would then learn DoS instead of BENIGN and score ~0.08
# accuracy rather than the ~0.83 that makes the accuracy-trap point.
# The test fold is never capped, so every reported metric is on the real
# distribution regardless of this setting.
TRAIN_CAP = 200_000


def train(cap: int | None = TRAIN_CAP, out_name: str = MODEL_NAME,
    score: bool = True, verbose: bool = True, **overrides) -> dict:
    """Fit XGBoost on the training fold and persist it to data/.

    `overrides` are passed through to models.build_xgboost, so the defaults can
    be changed without editing this module.

    Returns a stats dict: path, meta_path, rows, features, classes, cap,
    fit_seconds, and the test metrics when score=True.
    """
    t0 = time.time()
    split = sd.load_split(cap=cap)
    names = ev.class_names(split)

    X_train, y_train = split["X_train"], split["y_train"]
    if verbose:
        print(f"loaded in {time.time() - t0:.0f}s")
        print(f"training on {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
        counts = np.bincount(y_train, minlength=len(names))
        for i in np.argsort(-counts):
            print(f"  {names[i]:<12} {counts[i]:>10,}")

    model = md.build_xgboost(**overrides)
    t_fit = time.time()
    model.fit(X_train, y_train)
    fit_s = time.time() - t_fit

    path = bd.data_path(out_name)
    model.save_model(path)

    stats = {
        "rows": int(X_train.shape[0]),
        "features": int(X_train.shape[1]),
        "cap": cap,
        "classes": names,
        "feature_names": list(split["feature_names"]),
        "fit_seconds": round(fit_s, 1),
    }

    if score:
        pred = md.predict_in_chunks(model, split["X_test"])
        stats["test"] = ev.summary_metrics(split["y_test"], pred)
        stats["test_rows"] = int(len(split["y_test"]))

    meta_path = bd.data_path(META_NAME)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    stats["path"] = path
    stats["meta_path"] = meta_path

    if verbose:
        print(f"\nfit in {fit_s:.0f}s")
        if score:
            t = stats["test"]
            print(f"test macro-F1 {t['MacroF1']:.4f}   accuracy {t['Accuracy']:.4f}   "
                  f"({stats['test_rows']:,} rows)")
        print(f"saved {path}")
        print(f"saved {meta_path}")
        print(f"total {time.time() - t0:.0f}s")

    return stats


def load_model(name: str = MODEL_NAME) -> XGBClassifier:
    """Read a persisted model back.

    Predictions come back as class indices; pair them with load_metadata()
    ["classes"] to recover the labels.
    """
    model = XGBClassifier()
    model.load_model(bd.data_path(name))
    return model


def load_metadata(name: str = META_NAME) -> dict:
    """Class order, feature order and provenance of the persisted model."""
    with open(bd.data_path(name), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and persist the CIC-IDS2017 XGBoost model."
    )
    parser.add_argument(
        "--cap", type=int, default=TRAIN_CAP,
        help=f"BENIGN rows kept in the training fold (default {TRAIN_CAP:,}); "
             "pass 0 to train on the uncapped fold",
    )
    parser.add_argument("--out", default=MODEL_NAME, help="model filename in data/")
    parser.add_argument("--no-score", action="store_true",
                        help="skip the post-fit metrics on the test fold")
    args = parser.parse_args()

    cap = None if args.cap <= 0 else args.cap
    train(cap=cap, out_name=args.out, score=not args.no_score)


if __name__ == "__main__":
    main()
