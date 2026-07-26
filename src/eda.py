"""
EDA utilities for CIC-IDS2017 analysis.

Every function answers questions about data that will be used during model creation. 

Class balance:
    class_distribution() | grouped label counts + percentages on the full dataset
    imbalance_ratio() | largest class and smallest class comparisons
    plot_class_distribution() | horizontal bar chart on log axis of class distribution

Sampling:
    stratified_sample() | pre-class reproducible sample of the full dataset

Feature scale and shape:
    scale_summary() | min/median/max/std/range/skew per feature
    plot_feature_distributions() | log1p histograms of selected features

Data quality:
    negative_value_report() | columns holding physically impossible negatives
    impossible_value_rows() | corrupted rows displayed without sentinel values
    sentinel_report() | percentage of rows with a suspected sentinel value
    port_profile() | DestinationPort as an ordinal quantity check

Class-conditional structure:
    class_medians() | median of selected features by class
    plot_classwise_boxplots() | log1p distribution per class

Redundancy:
    correlated_pairs() | feature pairs above a correlation threshold
    identical_feature_pairs() | finds identical duplicate features
    plot_correlation_heatmap() | correlation matrix of features

Counts are computed on the FULL dataset.
Distribution shape and correlation are computed on athe stratified sample.

Every figure shares one visual system - see the "Plot styling".
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_dataset as bd

# Config
TARGET = bd.TARGET
GROUP_COL = bd.GROUP_COL
SOURCE_COL = bd.SOURCE_COL

FINAL_PATH = bd.data_path(bd.FINAL_NAME)
RESULTS_DIR = bd.data_path("results")

NON_FEATURE_COLS = (TARGET, GROUP_COL, SOURCE_COL)

BATCH_SIZE = bd.BATCH_SIZE
SAMPLE_CAP = 8_000
RANDOM_STATE = 42

# CIC-FlowMeter writes -1 into the initial-window when no value observed
SENTINEL_COLS = ("InitWinBytesForward", "InitWinBytesBackward")
SENTINEL_VALUE = -1.0

# Column values that can not be negative
NONNEGATIVE_COLS = (
    "FlowDuration", "FlowBytesS", "FlowPacketsS",
    "FlowIatMean", "FlowIatMax", "FlowIatMin", "FwdIatMin",
    "FwdHeaderLength", "BwdHeaderLength", "MinSegSizeForward",
)


# Plot styling
PRIMARY = "#3f7cac"
MUTED = "#9aa5b1"
ACCENT = "#d9822b"

GRID_COLOR = "0.85"
EDGE_COLOR = "0.35"
WHISKER_COLOR = "0.55"
SUBTITLE_COLOR = "0.35"

TITLE_SIZE = 13
SUBTITLE_SIZE = 9.5
LABEL_SIZE = 10
PANEL_TITLE_SIZE = 10.5
TICK_SIZE = 9
FIG_DPI = 150


# Helper functions
def results_path(*parts: str) -> str:
    """Normalised path inside data/results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.normpath(os.path.join(RESULTS_DIR, *parts))


def class_colors(labels: Sequence[str]) -> list[str]:
    """Grey for BENIGN, primary blue for every attack family."""
    return [MUTED if str(l).upper() == "BENIGN" else PRIMARY for l in labels]


def _style_axes(ax, grid_axis: str | None = "x",
    hide: Sequence[str] = ("top", "right", "left"),
    flat_ticks: str | None = "y",) -> None:
    """Light grid behind the data, minimal spines, no tick stubs."""
    if grid_axis:
        ax.grid(axis=grid_axis, which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    for side in hide:
        ax.spines[side].set_visible(False)
    if flat_ticks:
        ax.tick_params(axis=flat_ticks, length=0)


def _add_titles(ax, title: str, subtitle: str | None = None,
    gap_pt: float = 10, line_pt: float = 14) -> None:
    """Left-aligned title with the finding itself as a grey subtitle.

    Offsets are in points from the top-left of the axes to maintain spacing.
    """
    common = dict(xy=(0, 1), xycoords="axes fraction",
                  textcoords="offset points", ha="left", va="bottom")

    if subtitle:
        ax.annotate(subtitle, xytext=(0, gap_pt), fontsize=SUBTITLE_SIZE,
                    color=SUBTITLE_COLOR, **common)
        ax.annotate(title, xytext=(0, gap_pt + line_pt), fontsize=TITLE_SIZE, **common)
    else:
        ax.annotate(title, xytext=(0, gap_pt), fontsize=TITLE_SIZE, **common)


def _add_fig_titles(fig, title: str, subtitle: str | None = None,
    gap_in: float = 0.40, line_in: float = 0.26) -> None:
    """Add titles to the top-left of a figure, above the axes."""
    axes = [a for a in fig.axes if a.get_visible()]
    left = min(a.get_position().x0 for a in axes)
    top = max(a.get_position().y1 for a in axes)
    height = fig.get_figheight()

    if subtitle:
        fig.text(left, top + (gap_in + line_in) / height, title,
                 fontsize=TITLE_SIZE, ha="left", va="bottom")
        fig.text(left, top + gap_in / height, subtitle,
                 fontsize=SUBTITLE_SIZE, color=SUBTITLE_COLOR, ha="left", va="bottom")
    else:
        fig.text(left, top + gap_in / height, title,
                 fontsize=TITLE_SIZE, ha="left", va="bottom")


def _save(fig, save_name: str, tight: bool = True) -> str:
    """Write the figure to data/results and close"""
    import matplotlib.pyplot as plt

    if tight:
        fig.tight_layout()
    out = results_path(save_name)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _thousands(ax, axis: str = "x") -> None:
    """Format tick label numbers with commas."""
    import matplotlib.ticker as mticker

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    target.set_minor_formatter(mticker.NullFormatter())


def feature_columns(path: str = FINAL_PATH) -> list[str]:
    """Modelling features present in the parquet"""
    cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in cols if c not in NON_FEATURE_COLS]


def label_groups(path: str = FINAL_PATH) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (groups, keep) for every row in the file where `groups` is the LableGroup
    and `keep` masks EXCLUDED rows"""
    raw = pq.read_table(path, columns=[TARGET]).column(0).to_numpy()

    # map the ~15 unique strings rather than the 2.5M rows
    uniq, inv = np.unique(raw, return_inverse=True)
    mapped = np.array([bd.LABEL_MAP.get(u, bd.EXCLUDED_TAG) for u in uniq])
    groups = mapped[inv]
    return groups, groups != bd.EXCLUDED_TAG


# 1. Class balance
def class_distribution(path: str = FINAL_PATH, decimals: int = 3) -> pd.DataFrame:
    """Row count and share of each modelling class, on the full dataset."""
    groups, keep = label_groups(path)
    counts = pd.Series(groups[keep]).value_counts()
    return (
        pd.DataFrame(
            {
                "Count": counts,
                "Percentage": (counts / counts.sum() * 100).round(decimals),
            }
        )
        .rename_axis(GROUP_COL)
        .reset_index()
    )


def imbalance_ratio(dist: pd.DataFrame | None = None, path: str = FINAL_PATH) -> float:
    """Majority class size divided by minority class size."""
    dist = class_distribution(path) if dist is None else dist
    return float(dist["Count"].max() / dist["Count"].min())


def plot_class_distribution(dist: pd.DataFrame | None = None,path: str = FINAL_PATH,
    save_name: str = "eda_class_distribution.png",) -> str:
    """Horizontal bar chart on a log axis."""
    import matplotlib.pyplot as plt

    dist = class_distribution(path) if dist is None else dist
    dist = dist.sort_values("Count")

    counts = dist["Count"].to_numpy()
    labels = dist[GROUP_COL].to_numpy()
    total = counts.sum()
    y = np.arange(len(counts))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(y, counts, color=class_colors(labels), height=0.72, zorder=3)

    ax.set_xscale("log")
    left = 10 ** np.floor(np.log10(counts.min()))
    right = counts.max() * 1.8  # Headroom
    ax.set_xlim(left, right)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    _thousands(ax, "x")
    _style_axes(ax, grid_axis="x")

    # Move label inside bar if there is no room
    span = np.log10(right) - np.log10(left)
    for yi, v in zip(y, counts):
        frac = (np.log10(v) - np.log10(left)) / span
        inside = frac > 0.55
        ax.annotate(
            f"{v:,}  ({v / total:.2%})",
            xy=(v, yi),
            xytext=(-6 if inside else 6, 0),
            textcoords="offset points",
            va="center",
            ha="right" if inside else "left",
            fontsize=TICK_SIZE,
            color="white" if inside else "0.25",
        )

    ax.set_xlabel("rows (log scale)", fontsize=LABEL_SIZE, labelpad=8)
    _add_titles(
        ax,
        "Class distribution",
        f"log axis - {labels[-1]} outnumbers {labels[0]} "
        f"{counts[-1] / counts[0]:,.0f}:1",
    )
    return _save(fig, save_name)


# 2. Reproducible stratified sample
def stratified_sample(path: str = FINAL_PATH, cap: int = SAMPLE_CAP,
    random_state: int = RANDOM_STATE, batch_size: int = BATCH_SIZE,) -> pd.DataFrame:
    """Draw at most `cap` rows per class. Row positions are chosen from the label column 
    first, then the file is streamed and only the selected rows are retained."""
    groups, keep = label_groups(path)

    rng = np.random.default_rng(random_state)
    selected = np.zeros(len(groups), dtype=bool)
    for cls in np.unique(groups[keep]):
        idx = np.flatnonzero(keep & (groups == cls))
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        selected[idx] = True

    parts: list[pd.DataFrame] = []
    row_ptr = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        mask = selected[row_ptr: row_ptr + len(chunk)]
        row_ptr += len(chunk)
        if mask.any():
            parts.append(chunk.loc[mask])

    sample = pd.concat(parts, ignore_index=True)
    # streaming preserves file order, so the mask lines up with the rows kept
    sample[GROUP_COL] = groups[selected]
    return sample


# 3. Feature scale and shape
def scale_summary(sample: pd.DataFrame, features: Sequence[str] | None = None,
    sort_by: str = "Range", top: int | None = None,) -> pd.DataFrame:
    """Per-feature min/median/max/std/range/skew starting with widest range."""
    features = list(sample.columns) if features is None else list(features)
    desc = sample[features].describe().T[["min", "50%", "max", "std"]]
    desc.columns = ["Min", "Median", "Max", "Std"]
    desc["Range"] = desc["Max"] - desc["Min"]
    desc["Skew"] = sample[features].skew()
    out = desc.sort_values(sort_by, ascending=False)
    return out.head(top) if top else out


def plot_feature_distributions(sample: pd.DataFrame, features: Sequence[str],
    save_name: str = "eda_feature_dists.png",) -> str:
    """log1p histograms on a linear axis"""
    import matplotlib.pyplot as plt

    features = [c for c in features if c in sample.columns]
    fig, axes = plt.subplots(
        1, len(features), figsize=(4.0 * len(features), 3.6), sharey=False
    )
    axes = np.atleast_1d(axes)

    for i, (ax, col) in enumerate(zip(axes, features)):
        v = sample[col].to_numpy()
        v = v[np.isfinite(v) & (v > 0)]
        logged = np.log1p(v)

        ax.hist(logged, bins=50, color=PRIMARY, edgecolor="white",
                linewidth=0.6, zorder=3)
        ax.axvline(np.median(logged), color=ACCENT, linewidth=1.4,
                   linestyle="--", zorder=4)

        _style_axes(ax, grid_axis="y", hide=("top", "right", "left"), flat_ticks="y")
        _thousands(ax, "y")
        ax.set_title(col, fontsize=PANEL_TITLE_SIZE, loc="left", pad=8)
        ax.set_xlabel("log1p(value)", fontsize=TICK_SIZE, labelpad=6)
        ax.tick_params(labelsize=TICK_SIZE)
        if i == 0:
            ax.set_ylabel("flows in sample", fontsize=TICK_SIZE, labelpad=6)

    fig.tight_layout()
    _add_fig_titles(
        fig,
        "Feature distributions are heavy-tailed",
        "log1p axis, positive values only - dashed line marks the median",
    )
    return _save(fig, save_name, tight=False)


# 4. Data quality issues that survive stage-1 cleaning
def negative_value_report(path: str = FINAL_PATH,
    features: Sequence[str] | None = None,) -> pd.DataFrame:
    """
    Count negative values per column on the full dataset, one column at a time.
    """
    features = feature_columns(path) if features is None else list(features)
    _, keep = label_groups(path)
    n_kept = int(keep.sum())

    rows = []
    for col in features:
        v = pq.read_table(path, columns=[col]).column(0).to_numpy()[keep]
        n_neg = int((v < 0).sum())
        if n_neg:
            rows.append(
                {
                    "Column": col,
                    "NNegative": n_neg,
                    "Percentage": round(100 * n_neg / n_kept, 4),
                    "Min": float(v.min()),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("NNegative", ascending=False)
        .reset_index(drop=True)
    )


def impossible_value_rows(path: str = FINAL_PATH,
    exclude: Sequence[str] = SENTINEL_COLS,) -> tuple[int, float]:
    """
    Rows carrying at least one negative value outside the sentinel columns.
    Returns (n_rows, percentage) separating corruption from sentinels.
    """
    _, keep = label_groups(path)
    n_kept = int(keep.sum())

    columns = [c for c in negative_value_report(path)["Column"] if c not in set(exclude)]

    affected = np.zeros(n_kept, dtype=bool)
    for col in columns:
        v = pq.read_table(path, columns=[col]).column(0).to_numpy()[keep]
        affected |= v < 0

    n_bad = int(affected.sum())
    return n_bad, round(100 * n_bad / n_kept, 4)


def sentinel_report(path: str = FINAL_PATH, columns: Sequence[str] = SENTINEL_COLS,
    value: float = SENTINEL_VALUE,) -> pd.DataFrame:
    """Share of rows sitting exactly on a suspected missing-value marker."""
    _, keep = label_groups(path)
    n_kept = int(keep.sum())

    rows = []
    for col in columns:
        v = pq.read_table(path, columns=[col]).column(0).to_numpy()[keep]
        n_hit = int((v == value).sum())
        rows.append(
            {
                "Column": col,
                "NAtSentinel": n_hit,
                "Percentage": round(100 * n_hit / n_kept, 2),
                "MinExcludingSentinel": float(v[v != value].min()) if n_hit < n_kept else np.nan,
            }
        )
    return pd.DataFrame(rows)


def port_profile(path: str = FINAL_PATH, col: str = "DestinationPort",
    top: int = 10,) -> tuple[int, pd.DataFrame]:
    """Return (n_distinct_ports, top-N port table) for the full dataset."""
    _, keep = label_groups(path)
    v = pq.read_table(path, columns=[col]).column(0).to_numpy()[keep]

    uniq, counts = np.unique(v, return_counts=True)
    order = np.argsort(-counts)[:top]
    table = pd.DataFrame(
        {
            "Port": uniq[order].astype(np.int64),
            "Count": counts[order],
            "Percentage": (100 * counts[order] / len(v)).round(2),
        }
    )
    return len(uniq), table


# 5. Class-conditional structure
def class_medians(sample: pd.DataFrame, features: Sequence[str],
    group_col: str = GROUP_COL,) -> pd.DataFrame:
    """Median of each feature within each class."""
    features = [c for c in features if c in sample.columns]
    return sample.groupby(group_col)[list(features)].median()


def plot_classwise_boxplots(sample: pd.DataFrame, features: Sequence[str],
    group_col: str = GROUP_COL, max_cols: int = 2,
    save_name: str = "eda_classwise_boxplots.png",) -> str:
    """log1p boxplot per class."""
    import matplotlib.pyplot as plt

    features = [c for c in features if c in sample.columns]
    classes = sorted(sample[group_col].unique())
    colors = class_colors(classes)

    ncols = min(len(features), max_cols)
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(features):]:      # trailing panels on a ragged grid
        ax.set_visible(False)

    for ax, col in zip(axes, features):
        data = [
            np.log1p(sample.loc[sample[group_col] == c, col].clip(lower=0))
            for c in classes
        ]
        bp = ax.boxplot(
            data,
            tick_labels=classes,
            showfliers=False,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color=ACCENT, linewidth=1.6),
            boxprops=dict(edgecolor=EDGE_COLOR, linewidth=0.9),
            whiskerprops=dict(color=WHISKER_COLOR, linewidth=0.9),
            capprops=dict(color=WHISKER_COLOR, linewidth=0.9),
            zorder=3,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        _style_axes(ax, grid_axis="y", hide=("top", "right", "left"), flat_ticks="both")
        ax.set_title(f"log1p({col})", fontsize=PANEL_TITLE_SIZE, loc="left", pad=8)
        ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.tight_layout()
    _add_fig_titles(
        fig,
        "Class-conditional distributions",
        "log1p axis, outliers hidden - amber line marks the median, "
        "separated boxes mean an easily separable class",
    )
    return _save(fig, save_name, tight=False)


# 6. Redundancy
def correlated_pairs(sample: pd.DataFrame, features: Sequence[str] | None = None,
    threshold: float = 0.95,) -> pd.DataFrame:
    """Feature pairs whose absolute correlation exceeds `threshold`."""
    features = feature_columns() if features is None else list(features)
    corr = sample[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = upper.stack().sort_values(ascending=False)
    pairs = pairs[pairs > threshold]
    return pd.DataFrame(
        {
            "FeatureA": [a for a, _ in pairs.index],
            "FeatureB": [b for _, b in pairs.index],
            "AbsCorr": pairs.to_numpy().round(4),
        }
    )


def identical_feature_pairs(sample: pd.DataFrame,
    features: Sequence[str] | None = None,) -> pd.DataFrame:
    """Feature pairs that are element-wise identical. Hashes each column first."""
    features = feature_columns() if features is None else list(features)

    buckets: dict[int, list[str]] = {}
    for col in features:
        key = hash(sample[col].to_numpy().tobytes())
        buckets.setdefault(key, []).append(col)

    rows = []
    for cols in buckets.values():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if np.array_equal(sample[cols[i]].to_numpy(), sample[cols[j]].to_numpy()):
                    rows.append({"FeatureA": cols[i], "FeatureB": cols[j]})
    return pd.DataFrame(rows, columns=["FeatureA", "FeatureB"])


def verify_identical_on_full(pairs: pd.DataFrame, path: str = FINAL_PATH) -> pd.DataFrame:
    """Re-check candidate duplicate pairs against every row on the full dataset."""
    _, keep = label_groups(path)

    out = []
    for _, row in pairs.iterrows():
        a = pq.read_table(path, columns=[row["FeatureA"]]).column(0).to_numpy()[keep]
        b = pq.read_table(path, columns=[row["FeatureB"]]).column(0).to_numpy()[keep]
        out.append(
            {
                "FeatureA": row["FeatureA"],
                "FeatureB": row["FeatureB"],
                "IdenticalOnFullData": bool(np.array_equal(a, b)),
            }
        )
    return pd.DataFrame(out)


def plot_correlation_heatmap(sample: pd.DataFrame, features: Sequence[str] | None = None,
    n_features: int = 24, save_name: str = "eda_correlation.png",) -> str:
    """Correlation matrix of the first `n_features` columns."""
    import matplotlib.pyplot as plt

    features = feature_columns() if features is None else list(features)
    sub = sample[features[:n_features]]
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(sub.columns)))
    ax.set_yticklabels(sub.columns, fontsize=6.5)

    # a heatmap has no data-ink to separate from its frame, so drop the frame
    _style_axes(ax, grid_axis=None, hide=("top", "right", "left", "bottom"),
                flat_ticks="both")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=TICK_SIZE, labelpad=8)
    cbar.ax.tick_params(labelsize=TICK_SIZE, length=0)
    cbar.outline.set_visible(False)

    n_strong = int(
        (corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)) > 0.95).sum().sum()
    )
    _add_titles(
        ax,
        f"Feature correlation (first {len(sub.columns)} of {len(features)} features)",
        f"red = positive, blue = negative - {n_strong} pairs above |r| = 0.95 in this block",
    )
    return _save(fig, save_name)


if __name__ == "__main__":
    dist = class_distribution()
    print(dist.to_string(index=False))
    print(f"\nImbalance ratio: {imbalance_ratio(dist):,.1f} : 1")

    sample = stratified_sample()
    print(f"\nSample: {len(sample):,} rows")
    print(sample[GROUP_COL].value_counts().to_string())
