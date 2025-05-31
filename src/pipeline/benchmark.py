"""End-to-end benchmark: fit models, forecast every series, score, tabulate.

The evaluation protocol mirrors the M5 validation window: the final
``horizon`` observations of each series are held out, models see only the
preceding history, and global models are trained strictly on the held-in
portion to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import SeriesPanel, TimeSeries
from src.data.preprocessing import classify_series
from src.evaluation.metrics import mase, rmsse, wql
from src.models.base import Forecaster
from src.utils import Config, get_logger

log = get_logger("pipeline")


def _train_panel(panel: SeriesPanel, horizon: int) -> SeriesPanel:
    """Histories only (drop the last ``horizon`` points) for global training."""
    histories = []
    for ts in panel:
        if len(ts) > horizon:
            histories.append(
                TimeSeries(
                    item_id=ts.item_id,
                    values=ts.values[:-horizon],
                    start_date=ts.start_date,
                    freq=ts.freq,
                    static=ts.static,
                )
            )
    return SeriesPanel(histories, name=f"{panel.name}_train", seasonality=panel.seasonality)


def run_benchmark(
    panel: SeriesPanel,
    models: list[Forecaster],
    cfg: Config,
) -> pd.DataFrame:
    """Return a tidy per-(series, model) results frame."""
    horizon = cfg.data.horizon
    season = cfg.data.get("seasonality", 7)
    quantile_levels = np.asarray(cfg.data.quantiles, dtype=np.float64)

    strat = cfg.get("stratification", Config())
    classify_kwargs = dict(
        adi_threshold=strat.get("adi_threshold", 1.32),
        cv2_threshold=strat.get("cv2_threshold", 0.49),
        volume_buckets=tuple(strat.get("volume_buckets", (0.0, 1.0, 5.0, 1e9))),
        volume_labels=tuple(strat.get("volume_labels", ("intermittent", "medium", "dense"))),
    )

    # Train global models once on the held-in histories.
    train_panel = _train_panel(panel, horizon)
    for model in models:
        if model.is_global:
            log.info("Fitting global model: %s", model.name)
            model.fit(train_panel)

    rows: list[dict] = []
    for ts in tqdm(panel, desc="series", unit="ts"):
        if len(ts) <= horizon:
            continue
        history, actual = ts.split(horizon)
        stats = classify_series(ts, **classify_kwargs)

        for model in models:
            fc = model.predict(history, horizon, quantile_levels)
            row = {
                "item_id": ts.item_id,
                "model": model.name,
                "mase": mase(actual, fc.point, history.values, season),
                "rmsse": rmsse(actual, fc.point, history.values, season),
                "wql": wql(actual, fc.quantiles, fc.quantile_levels),
                "sb_class": stats.sb_class,
                "volume_bucket": stats.volume_bucket,
                "mean_demand": stats.mean,
                "adi": stats.adi,
                "cv2": stats.cv2,
                "zero_frac": stats.zero_frac,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Scored %d series x %d models = %d rows", len(panel), len(models), len(df))
    return df
