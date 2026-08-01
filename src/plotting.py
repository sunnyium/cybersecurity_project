"""
Shared figure styling for the CIC-IDS2017 project.

Every figure the project produces is styled using these helper functions. 

Palette and type scale:
    PRIMARY / MUTED / ACCENT | defines an accent, neutral and highlight color
    class_colors() | grey for BENIGN, standard blue for every attack family

Axes:
    style_axes() | light grid behind the data, minimal spines, no tick stubs
    thousands() | comma-separated tick labels

Titles:
    add_titles() | left-aligned title with the finding itself as a subtitle
    add_fig_titles() | the same block above a multi-panel figure

Output:
    results_path() | normalised path inside data/results
    save_figure() | write to data/results at FIG_DPI

Titles include findings, while the subtitle explains the content of each
visualization.
"""

from __future__ import annotations

import os
from typing import Sequence

import build_dataset as bd

RESULTS_DIR = bd.data_path("results")

# Palette
PRIMARY = "#3f7cac"
MUTED = "#9aa5b1"
ACCENT = "#d9822b"

GRID_COLOR = "0.85"
EDGE_COLOR = "0.35"
WHISKER_COLOR = "0.55"
SUBTITLE_COLOR = "0.35"

# Type scale
TITLE_SIZE = 13
SUBTITLE_SIZE = 9.5
LABEL_SIZE = 10
PANEL_TITLE_SIZE = 10.5
TICK_SIZE = 9
FIG_DPI = 150


def results_path(*parts: str) -> str:
    """Normalised path inside data/results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.normpath(os.path.join(RESULTS_DIR, *parts))


def class_colors(labels: Sequence[str]) -> list[str]:
    """Grey for BENIGN, primary blue for every attack family."""
    return [MUTED if str(l).upper() == "BENIGN" else PRIMARY for l in labels]


def style_axes(ax, grid_axis: str | None = "x",
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


def add_titles(ax, title: str, subtitle: str | None = None,
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


def add_fig_titles(fig, title: str, subtitle: str | None = None,
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


def save_figure(fig, save_name: str, tight: bool = True) -> str:
    """Write the figure to data/results and close"""
    import matplotlib.pyplot as plt

    if tight:
        fig.tight_layout()
    out = results_path(save_name)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def thousands(ax, axis: str = "x") -> None:
    """Format tick label numbers with commas."""
    import matplotlib.ticker as mticker

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    target.set_minor_formatter(mticker.NullFormatter())
