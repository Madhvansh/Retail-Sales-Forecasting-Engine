"""Aggregate per-series results by demand regime.

The project's headline finding — no single model dominates; foundation models
win on dense SKUs while classical methods win on the intermittent tail — falls
out of these stratified tables, not the dataset-wide average.
"""

from __future__ import annotations

import pandas as pd


def overall_table(df: pd.DataFrame, metrics: tuple[str, ...] = ("mase", "wql", "rmsse")) -> pd.DataFrame:
    """Mean metric per model across all series."""
    return df.groupby("model")[list(metrics)].mean().sort_values("wql")


def stratified_table(
    df: pd.DataFrame,
    by: str = "volume_bucket",
    metric: str = "wql",
) -> pd.DataFrame:
    """Mean ``metric`` per model within each stratum (model x stratum grid)."""
    pivot = df.pivot_table(index="model", columns=by, values=metric, aggfunc="mean")
    return pivot


def winners(df: pd.DataFrame, by: str = "volume_bucket", metric: str = "wql") -> pd.DataFrame:
    """Best (lowest-error) model in each stratum with its score and runner-up gap."""
    pivot = stratified_table(df, by=by, metric=metric)
    out = []
    for stratum in pivot.columns:
        col = pivot[stratum].dropna().sort_values()
        if col.empty:
            continue
        best, best_val = col.index[0], col.iloc[0]
        runner_up_gap = (col.iloc[1] - best_val) if len(col) > 1 else float("nan")
        out.append(
            {
                by: stratum,
                "n_series": int((df[by] == stratum).sum() / df["model"].nunique()),
                "winner": best,
                f"{metric}": round(float(best_val), 4),
                "gap_to_2nd": round(float(runner_up_gap), 4),
            }
        )
    return pd.DataFrame(out)


def win_rate(df: pd.DataFrame, metric: str = "wql") -> pd.DataFrame:
    """Per-model share of series on which it is the single best model."""
    idx = df.groupby("item_id")[metric].idxmin()
    best = df.loc[idx, ["item_id", "model"]]
    rate = best["model"].value_counts(normalize=True).rename("win_rate")
    return rate.to_frame().reset_index(names="model")
