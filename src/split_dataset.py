"""
src/split_dataset.py - loads the merged CICIDS2017 parquet from
    build_dataset.py, then builds train/test split datasets.

Memory: In order to prevent running out of memory as a result of the large
    amount of data, the code begins by creating one large array upfront. It
    proceeds to fill the array column-by-column and splitting data using INDEX
    ranges.
Label grouping: reads GROUP_COL from the parquet when build_dataset has
    materialised it, and falls back to mapping the raw labels through
    build_dataset.LABEL_MAP when it has not. Either way EXCLUDED rows are
    dropped before the feature matrix is allocated, so the array is sized to
    the rows actually kept rather than trimmed afterwards.
Majority subsampling: available but OFF by default. Capping the majority class
    would discard ~1.5M benign flows, and benign is the heterogeneous class
    whose breadth is what keeps false alarms down; class weights rebalance the
    loss without throwing rows away. When a cap is passed it applies to the
    TRAINING fold only -- the test fold is never resampled, so reported metrics
    reflect the real distribution.
"""
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import build_dataset as bd

FINAL_PATH = bd.data_path(bd.FINAL_NAME)

NON_FEATURE_COLS = (bd.TARGET, bd.GROUP_COL, bd.SOURCE_COL)
TARGET = bd.TARGET
TEST_SIZE = 0.2
RANDOM_STATE = 42
# None keeps every training row: imbalance is handled at the model with class
# weights rather than by discarding benign traffic. Pass an explicit cap to
# subsample instead -- kept available so the two can be compared.
DEFAULT_TRAIN_CAP = None


def feature_columns(path: str = FINAL_PATH):
    cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in cols if c not in NON_FEATURE_COLS]


def grouped_labels(path: str = FINAL_PATH):
    """Return (grouped, keep), where keep masks out the EXCLUDED rows.
    Uses the materialised GROUP_COL if present, otherwise maps TARGET through
    LABEL_MAP."""
    names = pq.ParquetFile(path).schema_arrow.names

    if bd.GROUP_COL in names:
        grouped = pq.read_table(path, columns=[bd.GROUP_COL]).column(0).to_numpy()
    else:
        raw = pq.read_table(path, columns=[TARGET]).column(0).to_numpy()
        # map the ~15 unique strings, not the 2.5M rows
        uniq, inv = np.unique(raw, return_inverse=True)
        unmapped = [u for u in uniq if u not in bd.LABEL_MAP]
        if unmapped:
            print(f"warning: labels absent from LABEL_MAP, dropped: {unmapped}")
        mapped = np.array([bd.LABEL_MAP.get(u, bd.EXCLUDED_TAG) for u in uniq])
        grouped = mapped[inv]

    return grouped, grouped != bd.EXCLUDED_TAG


def load_Xy(path: str = FINAL_PATH):
    """Load the full feature matrix (float32) and integer-encoded target.
    Returns (X, y, label_encoder, feature_names). Fills X column-by-column
    to keep peak memory near the final array size."""
    feat = feature_columns(path)
    grouped, keep = grouped_labels(path)

    pf = pq.ParquetFile(path)
    X = np.empty((int(keep.sum()), len(feat)), dtype=np.float32)
    for j, c in enumerate(feat):
        X[:, j] = pf.read(columns=[c]).column(0).to_numpy()[keep]

    le = LabelEncoder()
    y = le.fit_transform(grouped[keep]).astype(np.int16)
    return X, y, le, feat


def stratified_split_indices(y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(sss.split(np.zeros(len(y), dtype=np.int8), y))
    return np.sort(tr), np.sort(te)


def subsample_majority_train(train_idx, y, cap=DEFAULT_TRAIN_CAP,
                             random_state=RANDOM_STATE):
    """Cap the majority class in the training index, keeping every other row.

    The majority class is read off the training fold rather than hardcoded, so
    the cap always lands on whichever class actually dominates. On the full
    CICIDS2017 merge that is BENIGN (~83% of rows); every attack row survives.

    Test index is never passed here, so the test set is never subsampled.
    """
    if cap is None:
        return train_idx

    counts = np.bincount(y[train_idx])
    majority = int(np.argmax(counts))
    if counts[majority] <= cap:
        return train_idx

    rng = np.random.default_rng(random_state)
    ytr = y[train_idx]
    kept = train_idx[ytr != majority]
    capped = rng.choice(train_idx[ytr == majority], size=cap, replace=False)
    return np.sort(np.concatenate([capped, kept]))


def load_split(path: str = FINAL_PATH, cap=DEFAULT_TRAIN_CAP):
    """Convenience: returns a dict with train/test arrays ready for a model.
    The training majority class is capped; test reflects the true
    distribution."""
    X, y, le, feat = load_Xy(path)
    tr, te = stratified_split_indices(y)
    tr = subsample_majority_train(tr, y, cap=cap)

    out = {
        "X_train": X[tr],
        "y_train": y[tr],
        "X_test": X[te],
        "y_test": y[te],
        "label_encoder": le,
        "feature_names": feat,
    }
    del X  # the split copies are built; drop the full matrix
    return out


if __name__ == "__main__":
    X, y, le, feat = load_Xy()
    print(f"X: {X.shape}, {X.nbytes / 1e9:.2f} GB")
    print(f"classes: {list(le.classes_)}\n")

    tr, te = stratified_split_indices(y)
    # default is uncapped; 200k shown alongside to make the tradeoff visible
    demo_cap = 200_000
    tr_capped = subsample_majority_train(tr, y, cap=demo_cap)

    before = np.bincount(y[tr], minlength=len(le.classes_))
    after = np.bincount(y[tr_capped], minlength=len(le.classes_))
    test = np.bincount(y[te], minlength=len(le.classes_))

    print(f"{'class':<12} {'train (default)':>16} {'cap=' + f'{demo_cap:,}':>14} {'test':>10}")
    for c in np.argsort(-before):
        flag = "  <- capped" if after[c] < before[c] else ""
        print(f"{le.classes_[c]:<12} {before[c]:>16,} {after[c]:>14,} {test[c]:>10,}{flag}")

    print(f"\ntrain {len(tr):,} rows (default, uncapped) "
          f"| {len(tr_capped):,} with cap={demo_cap:,} | test {len(te):,} rows")
