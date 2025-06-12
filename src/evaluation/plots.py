"""Figures for the benchmark report.

All functions take the tidy per-series results frame and write a PNG. They use
a non-interactive backend so they run headless in CI / containers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.stratify import stratified_table  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def plot_overall(df: pd.DataFrame, metric: str, out_path: Path) -> Path:
    means = df.groupby("model")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot.barh(ax=ax, color="#4477aa")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Overall {metric.upper()} by model (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_stratified(df: pd.DataFrame, by: str, metric: str, out_path: Path) -> Path:
    pivot = stratified_table(df, by=by, metric=metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.T.plot.bar(ax=ax)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel(by.replace("_", " "))
    ax.set_title(f"{metric.upper()} by {by.replace('_', ' ')} and model")
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_win_rate(df: pd.DataFrame, metric: str, out_path: Path) -> Path:
    idx = df.groupby("item_id")[metric].idxmin()
    best = df.loc[idx, "model"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(best.values, labels=best.index, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Share of series won (by {metric.upper()})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all(df: pd.DataFrame, fig_dir: str | Path, primary: str = "wql") -> list[Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_overall(df, primary, fig_dir / f"overall_{primary}.png"),
        plot_overall(df, "mase", fig_dir / "overall_mase.png"),
        plot_stratified(df, "volume_bucket", primary, fig_dir / f"{primary}_by_volume.png"),
        plot_stratified(df, "sb_class", primary, fig_dir / f"{primary}_by_sbclass.png"),
        plot_win_rate(df, primary, fig_dir / "win_rate.png"),
    ]
    return paths
