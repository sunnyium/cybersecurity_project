"""
src/dataset.py - loads the merged CICIDS 2017 parquet from build_dataset.py,
    then builds train/test split datasets.

Memory: In order to prevent running out of memory as a result of the large
    amount of data, the code begins by creating one large array upfront. It 
    proceeds to fill the array column-by-column and splitting data using INDEX
    ranges.
Label grouping: Uses build_dataset.LABEL_MAP to create label grouping column. 
    EXCLUDED labels are dropped before the feature matrix is allocated.
(OPTIONAL ARGUMENT) Majority class subsampling: A large percentage of the data 
    is BENIGN. We cap BENIGN in the training set to 100k rows while keeping all 
    attack rows. The test set is not subsampled, so the model will be evaluated 
    on realistic data."""

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

# Default cap for majority (BENIGN) in the training set
# None if no cap is wanted
DEFAULT_TRAIN_CAP = None


def feature_columns(path: str = FINAL_PATH):
    cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in cols if c not in NON_FEATURE_COLS]


def grouped_labels(path: str = FINAL_PATH):
    """Return (grouped, keep), where keep masks out the EXCLUDED rows.
    Uses GROUP_COL if available, otherwise use LABEL_MAP"""

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
    """Subsamples training dataset if a cap is provided. An array of indicies 
    for the training set is returned."""
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
    """Convenience sampling - returns a dictionary with train/test arrays.
    Test set reflects actual label distribution."""
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
    del X # Drop full matrix
    return out


if __name__ == "__main__":
    X, y, le, feat = load_Xy()
    print(f"X: {X.shape}, {X.nbytes / 1e9:.2f} GB")
    print(f"classes: {list(le.classes_)}\n")

    tr, te = stratified_split_indices(y)
    tr = subsample_majority_train(tr, y)   # follows DEFAULT_TRAIN_CAP

    train = np.bincount(y[tr], minlength=len(le.classes_))
    test = np.bincount(y[te], minlength=len(le.classes_))

    print(f"{'class':<12} {'train':>12} {'test':>12}")
    for c in np.argsort(-train):
        print(f"{le.classes_[c]:<12} {train[c]:>12,} {test[c]:>12,}")

    print(f"\ntrain {len(tr):,} rows   test {len(te):,} rows")