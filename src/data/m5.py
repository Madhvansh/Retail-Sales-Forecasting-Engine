"""Load the M5 (Walmart) competition data into :class:`SeriesPanel` objects.

M5 ships three wide CSVs. ``sales_train_validation.csv`` has one row per
``item_id`` x ``store_id`` with 1 913 daily columns (``d_1`` .. ``d_1913``).
We melt the day columns into a per-series vector and attach the categorical
hierarchy (item / dept / cat / store / state) as static features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import SeriesPanel, TimeSeries
from src.utils import get_logger

log = get_logger("data.m5")

_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5(
    raw_dir: str | Path = "data/raw",
    max_series: int | None = None,
    seasonality: int = 7,
) -> SeriesPanel:
    """Read M5 raw CSVs and return a panel of daily demand series."""
    raw_dir = Path(raw_dir)
    sales_path = raw_dir / "sales_train_validation.csv"
    calendar_path = raw_dir / "calendar.csv"

    if not sales_path.exists():
        raise FileNotFoundError(
            f"{sales_path} not found. Run `python -m scripts.download_data` first, "
            "or set data.source='synthetic' in the config."
        )

    log.info("Reading %s", sales_path.name)
    sales = pd.read_csv(sales_path)
    if max_series is not None:
        sales = sales.iloc[:max_series].copy()

    day_cols = [c for c in sales.columns if c.startswith("d_")]

    start_date: np.datetime64 | None = None
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
        first_day = day_cols[0]
        start_date = np.datetime64(
            calendar.loc[calendar["d"] == first_day, "date"].iloc[0]
        )

    series: list[TimeSeries] = []
    for _, row in sales.iterrows():
        values = row[day_cols].to_numpy(dtype=np.float64)
        static = {c: row[c] for c in _ID_COLS if c in sales.columns}
        series.append(
            TimeSeries(
                item_id=str(row.get("id", row.name)),
                values=values,
                start_date=start_date,
                freq="D",
                static=static,
            )
        )

    log.info("Loaded %d M5 series of length %d", len(series), len(day_cols))
    return SeriesPanel(series, name="m5", seasonality=seasonality)
