"""Common forecaster interface used by every baseline and model.

All models emit a :class:`Forecast` that carries both a point forecast and a
set of predictive quantiles, because inventory decisions are driven by the
quantiles (service levels), not the mean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.data.dataset import SeriesPanel, TimeSeries


@dataclass
class Forecast:
    """Forecast for a single series over a horizon ``H``.

    Attributes
    ----------
    item_id : str
    point : np.ndarray
        Point (median/mean) forecast, shape ``(H,)``.
    quantiles : np.ndarray
        Predictive quantiles, shape ``(Q, H)``.
    quantile_levels : np.ndarray
        The ``Q`` levels in ``(0, 1)`` matching ``quantiles`` rows.
    """

    item_id: str
    point: np.ndarray
    quantiles: np.ndarray
    quantile_levels: np.ndarray

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64).ravel()
        self.quantiles = np.atleast_2d(np.asarray(self.quantiles, dtype=np.float64))
        self.quantile_levels = np.asarray(self.quantile_levels, dtype=np.float64).ravel()
        if self.quantiles.shape[0] != self.quantile_levels.shape[0]:
            raise ValueError("quantiles rows must match number of quantile levels")


class Forecaster(ABC):
    """Abstract base class for all forecasting models."""

    #: Human-readable model name used in result tables.
    name: str = "base"
    #: Whether the model is trained globally on the whole panel.
    is_global: bool = False

    def fit(self, panel: SeriesPanel) -> Forecaster:  # noqa: D401
        """Optional global training step. Local models can ignore this."""
        return self

    @abstractmethod
    def predict(
        self,
        history: TimeSeries,
        horizon: int,
        quantile_levels: np.ndarray,
    ) -> Forecast:
        """Forecast ``horizon`` steps ahead for one series."""

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _quantiles_from_samples(
        samples: np.ndarray, quantile_levels: np.ndarray
    ) -> np.ndarray:
        """Empirical quantiles from a ``(num_samples, H)`` sample matrix."""
        q = np.quantile(samples, quantile_levels, axis=0)
        return np.clip(q, 0.0, None)

    @staticmethod
    def _quantiles_from_normal(
        mean: np.ndarray, std: np.ndarray, quantile_levels: np.ndarray
    ) -> np.ndarray:
        """Gaussian quantiles, clipped at zero (demand is non-negative)."""
        from scipy.stats import norm

        z = norm.ppf(quantile_levels)[:, None]
        q = mean[None, :] + z * np.maximum(std[None, :], 1e-8)
        return np.clip(q, 0.0, None)
