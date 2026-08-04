"""
Dataset construction utilities for the CIC-IDS2017 pipeline.

Stage 1 (raw CSV -> analysis-ready parquet):
    merge_files() | clean + concatenate the 8 daily CSVs -> merged_rawlabels.parquet
    clean_merged() | drop cross-file duplicates + constant columns, CamelCase the
        column names -> merge_complete.parquet
    build_dataset() | runs both of the above end to end

Stage 2 (label handling / EDA):
    label_by_file() | attack types present in each source file
    label_distribution() | label counts + percentages
    class_support_table() | label support against the exclusion threshold
    add_label_group() | collapse fine-grained labels into modelling groups
    load_dataset() | read merge_complete.parquet, optionally with LabelGroup applied
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

#Config
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

TARGET = "Label"
GROUP_COL = "LabelGroup"

#Source column is originally "Source File" and becomes "SourceFile" after CamelCase changes
RAW_SOURCE_COL = "Source File"
SOURCE_COL = "SourceFile"

MERGED_NAME = "merged_rawlabels.parquet"
FINAL_NAME = "merge_complete.parquet"

BATCH_SIZE = 250_000

# Files in chronological order
FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# Remove duplication of "Fwd Header Length" column
DUPLICATE_COLS = ("Fwd Header Length.1",)
EXCLUDED_TAG = "EXCLUDED"

# Columns whose values cannot legitimately be negative. A negative duration,
# rate or header length is counter overflow in CIC-FlowMeter, not a measurement,
# so those rows are dropped in stage 1.
#
# Named in their CamelCase form and matched through to_camel_case, so the check
# is written once against canonical names and does not depend on the raw
# spelling of each header. The InitWinBytes columns are deliberately absent:
# -1 there is a documented "not observed" sentinel, not corruption.
NONNEGATIVE_COLS = (
    "FlowDuration", "FlowBytesS", "FlowPacketsS",
    "FlowIatMean", "FlowIatMax", "FlowIatMin", "FwdIatMin",
    "FwdHeaderLength", "BwdHeaderLength", "MinSegSizeForward",
)

# Minimum raw rows for a label to be modelled at all. At TEST_SIZE=0.2 this
# leaves ~100 test instances; below that a per-class precision or recall moves
# in visible jumps and cannot be reported in good faith.
MIN_CLASS_SUPPORT = 500

# Labels are converted into their respective modelling groups.
# Infiltration (36 rows) and Heartbleed (11) fall far below MIN_CLASS_SUPPORT and
# are tagged for removal; class_support_table() states that criterion explicitly.
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": EXCLUDED_TAG,
    "Heartbleed": EXCLUDED_TAG
}


# Helper Functions
def data_path(*parts: str) -> str:
    """Normalized path inside DATA_DIR."""
    return os.path.normpath(os.path.join(DATA_DIR, *parts))


def to_camel_case(col: str) -> str:
    """'Fwd Packet Length Max' -> 'FwdPacketLengthMax'."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", col.strip()) if p]
    return "".join(p.capitalize() for p in parts)


def _finite_mask(df: pd.DataFrame, exclude: Iterable[str] = ()) -> np.ndarray:
    """Row mask that is True where every numeric column is finite."""
    exclude = set(exclude)
    mask = np.ones(len(df), dtype=bool)
    for c in df.columns:
        if c in exclude:
            continue
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.number):
            mask &= np.isfinite(v.astype(np.float64, copy=False))
    return mask


def _nonnegative_mask(df: pd.DataFrame,
    nonnegative_cols: Iterable[str] = NONNEGATIVE_COLS) -> np.ndarray:
    """Row mask that is True where no impossible negative value appears.

    Raw headers are matched by their CamelCase form, so this works on the
    pre-rename frame without hardcoding the original spellings.
    """
    targets = set(nonnegative_cols)
    mask = np.ones(len(df), dtype=bool)
    for c in df.columns:
        if to_camel_case(c) not in targets:
            continue
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.number):
            mask &= ~(v < 0)
    return mask


# 1. Schema verification
def read_schemas(files: Sequence[str] = FILE_ORDER, data_dir: str = DATA_DIR,
    drop_cols: Sequence[str] = DUPLICATE_COLS,) -> dict[str, tuple[str, ...]]:
    """Map each source file to its normalised column tuple"""
    drop = set(drop_cols)
    schemas: dict[str, tuple[str, ...]] = {}
    for file in files:
        cols = pd.read_csv(os.path.join(data_dir, file), nrows=0).columns.str.strip().tolist()
        schemas[file] = tuple(c for c in cols if c not in drop)
    return schemas


def n_distinct_schemas(schemas: dict[str, tuple[str, ...]] | None = None) -> int:
    """Number of distinct header layouts across the source files."""
    schemas = read_schemas() if schemas is None else schemas
    return len(set(schemas.values()))


# 2. Within file cleaning and merge separate files
def clean_file(df: pd.DataFrame, target: str = TARGET,
    drop_cols: Sequence[str] = DUPLICATE_COLS, downcast: bool = True,
    nonnegative_cols: Sequence[str] = NONNEGATIVE_COLS,
    stats: dict | None = None,) -> pd.DataFrame:
    """
    Clean a single raw day of traffic.

    - strips whitespace from headers and drops the repeated header column
    - repairs the mojibake dash in the Web Attack labels
    - drops rows carrying NaN/inf in any numeric feature
    - drops rows with an impossible negative duration, rate or header length
    - drops exact within-file duplicates
    - downcasts numeric columns to float32

    Pass a dict as `stats` to receive the per-stage row accounting.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df[target] = (
        df[target].astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
    )

    n_in = len(df)
    df = df.loc[_finite_mask(df, exclude={target})]
    n_finite = len(df)

    keep = _nonnegative_mask(df, nonnegative_cols)
    n_negative = int((~keep).sum())
    df = df.loc[keep]

    df = df.drop_duplicates().copy()

    if downcast:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype(np.float32)
        df = df.loc[_finite_mask(df, exclude={target})].copy()

    if stats is not None:
        stats.update(
            rows_in=n_in,
            dropped_nonfinite=n_in - n_finite,
            dropped_negative=n_negative,
            rows_out=len(df),
        )

    return df


def merge_files(files: Sequence[str] = FILE_ORDER, data_dir: str = DATA_DIR, 
    out_name: str = MERGED_NAME, source_col: str = RAW_SOURCE_COL, 
    verbose: bool = True,) -> tuple[str, dict[str, int]]:
    """
    Clean each CSV and stream it into a single parquet file.

    Returns (output_path, {file: n_rows_kept}).
    """
    out_path = data_path(out_name)
    writer = None
    schema_ref = None
    counts: dict[str, int] = {}
    n_negative = 0

    try:
        for file in files:
            df = pd.read_csv(os.path.join(data_dir, file), low_memory=False)
            file_stats: dict[str, int] = {}
            df = clean_file(df, stats=file_stats)
            df[source_col] = file

            n_negative += file_stats.get("dropped_negative", 0)
            counts[file] = len(df)
            if verbose:
                print(f"{file}: {len(df):,} rows")

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema_ref = table.schema
                writer = pq.ParquetWriter(out_path, schema_ref)
            else:
                table = table.cast(schema_ref)
            writer.write_table(table)
            del df, table
    finally:
        if writer is not None:
            writer.close()

    if verbose:
        print(f"\nRows dropped by the sign check: {n_negative:,}")

    return out_path, counts


# 3. Cross-file cleaning, constant-column removal, and column renaming
def scan_duplicates_and_constants(path: str, target: str = TARGET, 
    ignore_cols: Sequence[str] = (RAW_SOURCE_COL,), batch_size: int = BATCH_SIZE,
    ) -> tuple[np.ndarray, list[str]]:
    """
    Single streaming pass over `path` that returns

    - dup_mask: True for rows whose (features, label) tuple was already seen
    - constant_cols: feature columns with a single distinct value

    `ignore_cols` is excluded from the duplicate key, so provenance does not
    mask genuine cross-file duplicates.
    """
    ignore = set(ignore_cols)
    pf = pq.ParquetFile(path)
    all_cols = list(pf.schema_arrow.names)

    key_cols = [c for c in all_cols if c not in ignore] # features + label
    cand_cols = [c for c in key_cols if c != target] # features only

    seen: set[int] = set()
    flag_chunks: list[np.ndarray] = []
    uniques: dict[str, set] = {c: set() for c in cand_cols}

    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()

        h = pd.util.hash_pandas_object(chunk[key_cols], index=False).to_numpy()
        flags = np.fromiter(
            ((v in seen) or bool(seen.add(v)) for v in h), dtype=bool, count=len(h)
        )
        flag_chunks.append(flags)

        for c in cand_cols:
            if len(uniques[c]) <= 1:
                # three distinct values is more than enough to rule out constancy
                uniques[c].update(pd.unique(chunk[c])[:3].tolist())

    dup_mask = (
        np.concatenate(flag_chunks) if flag_chunks else np.zeros(0, dtype=bool)
    )
    constant_cols = [c for c, u in uniques.items() if len(u) <= 1]
    return dup_mask, constant_cols


def clean_merged(src_name: str = MERGED_NAME, out_name: str = FINAL_NAME, 
    camel_case: bool = True, batch_size: int = BATCH_SIZE, verbose: bool = True,) -> dict:
    """
    Drop cross-file duplicates and constant columns, optionally renaming the
    surviving columns to CamelCase on the way out.

    Returns a stats dict: path, n_duplicates, constant_cols, rows_out, columns.
    """
    src = data_path(src_name)
    dst = data_path(out_name)

    dup_mask, constant_cols = scan_duplicates_and_constants(
        src, batch_size=batch_size
    )

    all_cols = list(pq.ParquetFile(src).schema_arrow.names)
    keep_cols = [c for c in all_cols if c not in set(constant_cols)]
    out_names = [to_camel_case(c) for c in keep_cols] if camel_case else list(keep_cols)

    writer = None
    row_ptr = 0
    rows_out = 0

    try:
        for batch in pq.ParquetFile(src).iter_batches(
            batch_size=batch_size, columns=keep_cols
        ):
            chunk = batch.to_pandas()
            m = ~dup_mask[row_ptr: row_ptr + len(chunk)]
            row_ptr += len(chunk)

            out = chunk.loc[m]
            rows_out += len(out)

            table = pa.Table.from_pandas(out, preserve_index=False)
            table = table.rename_columns(out_names)
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    stats = {
        "path": dst,
        "n_duplicates": int(dup_mask.sum()),
        "constant_cols": constant_cols,
        "rows_out": rows_out,
        "columns": out_names,
    }

    if verbose:
        print(f"Number of inter-file duplicates: {stats['n_duplicates']:,}")
        print(f"Number of constant columns: {len(constant_cols)}")
        print(f"Rows written: {rows_out:,}")

    return stats


def build_dataset(verbose: bool = True) -> dict:
    """Run the full stage-1 pipeline: raw CSVs -> merge_complete.parquet."""
    merged_path, counts = merge_files(verbose=verbose)
    if verbose:
        print(f"\nMerged to path: {merged_path}\n")

    stats = clean_merged(verbose=verbose)
    stats["per_file_counts"] = counts
    if verbose:
        print(f"\nWritten to path: {stats['path']}")
    return stats

# 4. Categorizing attack type labels
def load_dataset(path: str | None = None, columns: Sequence[str] | None = None, 
    label_group: bool = False, drop_excluded: bool = True,) -> pd.DataFrame:
    """Read merge_complete.parquet, optionally attaching the LabelGroup column."""
    path = data_path(FINAL_NAME) if path is None else path
    cols = list(columns) if columns is not None else None
    if cols is not None and label_group and TARGET not in cols:
        cols = cols + [TARGET]

    df = pq.read_table(path, columns=cols).to_pandas()
    if label_group:
        df = add_label_group(df, drop_excluded=drop_excluded)
    return df


def label_by_file(path: str | None = None, files: Sequence[str] = FILE_ORDER, 
    source_col: str = SOURCE_COL, target: str = TARGET,) -> pd.DataFrame:
    """One row per source file: row count and the attack types it contains."""
    path = data_path(FINAL_NAME) if path is None else path
    df = pq.read_table(path, columns=[target, source_col]).to_pandas()

    rows = []
    for file in files:
        m = df[source_col] == file
        rows.append(
            {
                "file": file,
                "n_rows": int(m.sum()),
                "attack_types": ", ".join(sorted(df.loc[m, target].unique())),
            }
        )
    return pd.DataFrame(rows)


def label_distribution(path: str | None = None, target: str = TARGET, 
    decimals: int = 2,) -> pd.DataFrame:
    """Count and percentage share of every label."""
    path = data_path(FINAL_NAME) if path is None else path
    labels = pq.read_table(path, columns=[target]).to_pandas()[target]

    counts = labels.value_counts()
    return (
        pd.DataFrame(
            {
                "Count": counts,
                "Percentage": (counts / len(labels) * 100).round(decimals),
            }
        )
        .rename_axis(target)
        .reset_index()
    )


def class_support_table(path: str | None = None, label_map: dict[str, str] = LABEL_MAP,
    target: str = TARGET, min_support: int = MIN_CLASS_SUPPORT,
    test_size: float = 0.2,) -> pd.DataFrame:
    """Raw label support measured against the exclusion threshold.

    States the exclusion criterion instead of leaving it implicit in LABEL_MAP.
    A class has to survive a stratified split and still leave enough test rows
    to carry a per-class metric; `min_support` rows is the floor, which at a
    20% test share is ~100 test instances.

    Returns one row per raw label with its count, the test rows it would
    contribute, whether it clears the floor, and the resulting decision.
    """
    tbl = label_distribution(path=path, target=target).copy()
    tbl["LabelGroup"] = tbl[target].map(label_map)
    tbl["TestRowsAtSplit"] = (tbl["Count"] * test_size).astype(int)
    tbl["MeetsMinimum"] = tbl["Count"] >= min_support
    tbl["Decision"] = np.where(tbl["LabelGroup"] == EXCLUDED_TAG, "EXCLUDE", "keep")

    return tbl[[target, "LabelGroup", "Count", "Percentage",
                "TestRowsAtSplit", "MeetsMinimum", "Decision"]]


def add_label_group(df: pd.DataFrame, label_map: dict[str, str] = LABEL_MAP,
    target: str = TARGET, group_col: str = GROUP_COL, drop_excluded: bool = True,
    ) -> pd.DataFrame:
    """
    Collapse fine-grained labels into modelling groups.

    Unmapped labels raise a warning and are left as NaN rather than silently
    disappearing. With drop_excluded=True the rare classes tagged EXCLUDED
    (Infiltration, Heartbleed) are removed.
    """
    df = df.copy()

    unknown = sorted(set(df[target].unique()) - set(label_map))
    if unknown:
        warnings.warn(f"Labels missing from LABEL_MAP, mapped to NaN: {unknown}")

    df[group_col] = df[target].map(label_map)
    if drop_excluded:
        df = df.loc[df[group_col] != EXCLUDED_TAG].reset_index(drop=True)
    return df


if __name__ == "__main__":
    build_dataset()
