"""Croston-family methods for intermittent demand.

Implements classic Croston, the Syntetos-Boylan Approximation (SBA) and
Teunter-Syntetos-Babai (TSB). Point forecasts are the usual constant
level/probability estimates; predictive quantiles come from a parametric
bootstrap (Bernoulli occurrence x resampled non-zero sizes), which respects
the zero-inflated nature of the demand and is what the WQL metric rewards.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import TimeSeries
from src.models.base import Forecast, Forecaster


class Croston(Forecaster):
    name = "croston"
    is_global = False

    def __init__(
        self,
        variant: str = "sba",
        alpha: float = 0.1,
        num_samples: int = 500,
        seed: int = 0,
    ) -> None:
        if variant not in {"classic", "sba", "tsb"}:
            raise ValueError(f"unknown Croston variant {variant!r}")
        self.variant = variant
        self.alpha = alpha
        self.num_samples = num_samples
        self.seed = seed
        self.name = f"croston_{variant}"

    def _fit_level_interval(self, y: np.ndarray) -> tuple[float, float, float]:
        """Return (size_level z, interval p, demand_probability)."""
        nz_idx = np.flatnonzero(y > 0)
        if nz_idx.size == 0:
            return 0.0, float(len(y)) or 1.0, 0.0

        sizes = y[nz_idx]
        intervals = np.diff(np.concatenate([[-1], nz_idx]))  # gaps incl. first

        a = self.alpha
        z = sizes[0]
        p = intervals[0]
        for s, q in zip(sizes[1:], intervals[1:]):
            z = a * s + (1 - a) * z
            p = a * q + (1 - a) * p
        prob = 1.0 / p if p > 0 else 0.0
        return float(z), float(max(p, 1.0)), float(prob)

    def _point(self, z: float, p: float, prob: float) -> float:
        if self.variant == "classic":
            return z / p if p > 0 else 0.0
        if self.variant == "sba":
            return (1 - self.alpha / 2.0) * (z / p) if p > 0 else 0.0
        # tsb: probability-based, robust to obsolescence
        return prob * z

    def predict(
        self,
        history: TimeSeries,
        horizon: int,
        quantile_levels: np.ndarray,
    ) -> Forecast:
        y = history.values
        z, p, prob = self._fit_level_interval(y)
        point_val = self._point(z, p, prob)
        point = np.full(horizon, max(point_val, 0.0))

        rng = np.random.default_rng(self.seed)
        nz_sizes = y[y > 0]
        if nz_sizes.size == 0:
            quantiles = np.zeros((len(quantile_levels), horizon))
            return Forecast(history.item_id, point, quantiles, quantile_levels)

        # Parametric bootstrap of future demand paths.
        occur = rng.random((self.num_samples, horizon)) < max(min(prob, 1.0), 0.0)
        sampled_sizes = rng.choice(nz_sizes, size=(self.num_samples, horizon), replace=True)
        paths = occur * sampled_sizes

        quantiles = self._quantiles_from_samples(paths, quantile_levels)
        return Forecast(
            item_id=history.item_id,
            point=point,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
        )
