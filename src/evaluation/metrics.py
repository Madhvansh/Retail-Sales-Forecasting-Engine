"""Forecast accuracy metrics.

Point metrics are *scaled* (MASE, RMSSE) so they are comparable across series
with very different demand volumes — essential when aggregating across the M5
catalogue.
"""

from __future__ import annotations

import numpy as np


def _seasonal_naive_scale(history: np.ndarray, season_length: int) -> float:
    """In-sample mean absolute seasonal difference (the MASE denominator)."""
    if len(history) <= season_length:
        denom = np.mean(np.abs(np.diff(history))) if len(history) > 1 else 0.0
    else:
        denom = np.mean(np.abs(history[season_length:] - history[:-season_length]))
    # Guard against degenerate (constant) histories.
    return float(denom) if denom > 1e-8 else 1e-8


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
    season_length: int = 7,
) -> float:
    """Mean Absolute Scaled Error."""
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    scale = _seasonal_naive_scale(np.asarray(history, dtype=np.float64), season_length)
    return float(np.mean(np.abs(actual - forecast)) / scale)


def rmsse(
    actual: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
    season_length: int = 7,
) -> float:
    """Root Mean Squared Scaled Error (the M5 'Accuracy' track metric)."""
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64)
    if len(hist) > season_length:
        denom = np.mean((hist[season_length:] - hist[:-season_length]) ** 2)
    else:
        denom = np.mean(np.diff(hist) ** 2) if len(hist) > 1 else 0.0
    denom = denom if denom > 1e-8 else 1e-8
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / denom))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


# --------------------------------------------------------------------------- #
# Probabilistic metrics                                                       #
# --------------------------------------------------------------------------- #
def quantile_loss(
    actual: np.ndarray,
    quantile_forecast: np.ndarray,
    quantile_levels: np.ndarray,
) -> np.ndarray:
    """Pinball loss per quantile.

    Parameters
    ----------
    actual : (H,) array
    quantile_forecast : (Q, H) array
    quantile_levels : (Q,) array

    Returns
    -------
    (Q,) array of summed pinball losses per quantile level.
    """
    actual = np.asarray(actual, dtype=np.float64)[None, :]
    q = np.asarray(quantile_forecast, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)[:, None]
    err = actual - q
    loss = np.maximum(levels * err, (levels - 1.0) * err)
    return loss.sum(axis=1)


def wql(
    actual: np.ndarray,
    quantile_forecast: np.ndarray,
    quantile_levels: np.ndarray,
    scale: float | None = None,
) -> float:
    """Weighted Quantile Loss.

    The total pinball loss across quantiles and horizon, normalised by the sum
    of absolute actuals (or an explicit ``scale``). This is the GluonTS/Chronos
    definition of WQL and is scale-free, so it can be averaged across series.
    """
    pinball = quantile_loss(actual, quantile_forecast, quantile_levels)
    total = 2.0 * pinball.sum()  # factor 2 matches the GluonTS convention
    denom = scale if scale is not None else float(np.sum(np.abs(actual)))
    denom = denom if denom > 1e-8 else 1e-8
    return float(total / denom)


def coverage(
    actual: np.ndarray,
    quantile_forecast: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[float, float]:
    """Empirical coverage: fraction of actuals at or below each quantile."""
    actual = np.asarray(actual, dtype=np.float64)
    q = np.asarray(quantile_forecast, dtype=np.float64)
    return {
        float(level): float(np.mean(actual <= q[i]))
        for i, level in enumerate(np.asarray(quantile_levels))
    }
