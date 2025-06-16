"""Streamlit dashboard for exploring benchmark results.

Run with::

    streamlit run app/dashboard.py

It reads ``results/metrics.csv`` (produced by ``scripts.run_benchmark``) and
lets you slice WQL/MASE by demand regime and inspect per-model win rates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.stratify import overall_table, stratified_table, win_rate, winners

st.set_page_config(page_title="Retail Forecasting Benchmark", layout="wide")

RESULTS = Path("results/metrics.csv")


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Retail Demand Forecasting — Foundation vs. Supervised vs. Classical")
st.caption("Chronos-Bolt (zero-shot) vs. PatchTST vs. seasonal-naive / Croston on M5.")

results_path = st.sidebar.text_input("Results CSV", str(RESULTS))
metric = st.sidebar.selectbox("Metric", ["wql", "mase", "rmsse"], index=0)
by = st.sidebar.selectbox("Stratify by", ["volume_bucket", "sb_class"], index=0)

if not Path(results_path).exists():
    st.warning(
        f"No results at `{results_path}`. Run "
        "`python -m scripts.run_benchmark` first."
    )
    st.stop()

df = load(results_path)

col1, col2, col3 = st.columns(3)
col1.metric("Series", df["item_id"].nunique())
col2.metric("Models", df["model"].nunique())
best_overall = overall_table(df)[metric].idxmin()
col3.metric(f"Best overall ({metric.upper()})", best_overall)

st.subheader("Overall ranking")
st.dataframe(overall_table(df).round(4), use_container_width=True)

st.subheader(f"{metric.upper()} by {by.replace('_', ' ')}")
pivot = stratified_table(df, by=by, metric=metric)
st.bar_chart(pivot.T)
st.dataframe(pivot.round(4), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Winner per stratum")
    st.dataframe(winners(df, by=by, metric=metric), use_container_width=True)
with right:
    st.subheader("Per-model win rate")
    st.dataframe(win_rate(df, metric=metric), use_container_width=True)

st.subheader("Demand-regime landscape (ADI vs. CV²)")
scatter_df = df.drop_duplicates("item_id")[["adi", "cv2", "sb_class", "mean_demand"]]
st.scatter_chart(scatter_df, x="adi", y="cv2", color="sb_class", size="mean_demand")
