"""
Check for memory availabiltiy
Split dataset by reading parquet, stacking features, and train_test_split
"""
import time
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import build_dataset as bd

PROC = psutil.Process()
GB = 1024 ** 3


def rss_gb() -> float:
    """Memory held by process"""
    return PROC.memory_info().rss / GB


def peak_gb() -> float | None:
    info = PROC.memory_info()
    return getattr(info, "peak_wset", None) and info.peak_wset / GB


def step(label: str, t0: float, baseline: float) -> float:
    now = rss_gb()
    print(f"{label:<28} {now:6.2f} GB  (+{now - baseline:5.2f})  {time.time() - t0:5.1f}s")
    return now


def main() -> None:
    t0 = time.time()
    base = rss_gb()
    print(f"{'start':<28} {base:6.2f} GB")

    df = pd.read_parquet(bd.data_path(bd.FINAL_NAME))
    prev = step("read_parquet", t0, base)

    df = bd.add_label_group(df, drop_excluded=True)
    prev = step("add_label_group (.copy)", t0, prev)

    feat = [c for c in df.columns if c not in (bd.TARGET, bd.GROUP_COL, bd.SOURCE_COL)]

    X = np.column_stack([df[c].to_numpy(np.float32) for c in feat])
    prev = step("column_stack", t0, prev)

    le = LabelEncoder()
    y = le.fit_transform(df[bd.GROUP_COL]).astype(np.int16)
    prev = step("label encode", t0, prev)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    step("train_test_split", t0, prev)

    print(f"\nX          {X.shape}  {X.nbytes / GB:.2f} GB")
    print(f"X_train    {X_train.shape}")
    print(f"X_test     {X_test.shape}")
    print(f"classes    {list(le.classes_)}")

    peak = peak_gb()
    if peak is not None:
        print(f"\npeak working set: {peak:.2f} GB  vs {X.nbytes / GB:.2f} GB of actual data")


if __name__ == "__main__":
    main()
