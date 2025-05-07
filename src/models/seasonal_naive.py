"""Seasonal-naive baseline.

The point forecast repeats the last seasonal cycle. Predictive quantiles are
formed from the empirical distribution of in-sample seasonal residuals, which
gives a surprisingly strong probabilistic baseline on dense retail series.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import TimeSeries
from src.models.base import Forecast, Forecaster


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"
    is_global = False

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = max(1, int(season_length))

    def predict(
        self,
        history: TimeSeries,
        horizon: int,
        quantile_levels: np.ndarray,
    ) -> Forecast:
        y = history.values
        m = self.season_length

        if len(y) < m:
            point = np.full(horizon, float(y.mean()) if len(y) else 0.0)
        else:
            last_season = y[-m:]
            reps = int(np.ceil(horizon / m))
            point = np.tile(last_season, reps)[:horizon]

        # In-sample seasonal residuals -> empirical predictive distribution.
        if len(y) > m:
            resid = y[m:] - y[:-m]
        else:
            resid = np.array([0.0])

        # Build quantiles by adding residual quantiles to the point forecast,
        # scaling the spread by sqrt(h) to reflect growing uncertainty.
        resid_q = np.quantile(resid, quantile_levels)
        steps = np.sqrt(np.arange(1, horizon + 1))
        quantiles = point[None, :] + resid_q[:, None] * steps[None, :]
        quantiles = np.clip(quantiles, 0.0, None)

        return Forecast(
            item_id=history.item_id,
            point=np.clip(point, 0.0, None),
            quantiles=quantiles,
            quantile_levels=quantile_levels,
        )
