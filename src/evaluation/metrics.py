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
