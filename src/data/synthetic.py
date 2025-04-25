"""Synthetic M5-like demand generator.

Produces a realistic mix of demand profiles so the full pipeline runs without
the (large) Kaggle download. The mix deliberately spans the
Syntetos-Boylan quadrants — smooth/dense, erratic, intermittent and lumpy —
because the headline finding of this project is that model ranking flips
across these regimes.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import SeriesPanel, TimeSeries

_PROFILES = ("dense", "erratic", "intermittent", "lumpy")
_CATEGORIES = ("FOODS", "HOUSEHOLD", "HOBBIES")
_STORES = ("CA_1", "CA_2", "TX_1", "WI_1")


def _weekly_seasonality(length: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(length)
    return amplitude * (1.0 + 0.5 * np.sin(2 * np.pi * t / 7.0 + phase))


def _make_series(idx: int, length: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    if profile == "dense":
        base = rng.uniform(8, 40)
        season = _weekly_seasonality(length, base, rng)
        trend = np.linspace(0, rng.uniform(-0.2, 0.4) * base, length)
        lam = np.clip(season + trend, 0.1, None)
        return rng.poisson(lam).astype(np.float64)

    if profile == "erratic":
        base = rng.uniform(3, 12)
        season = _weekly_seasonality(length, base, rng)
        noise = rng.gamma(shape=1.2, scale=base / 2, size=length)
        return np.round(np.clip(season * 0.4 + noise, 0, None)).astype(np.float64)

    if profile == "intermittent":
        p = rng.uniform(0.08, 0.25)  # probability of a demand event
        occur = rng.random(length) < p
        sizes = rng.poisson(rng.uniform(1.0, 3.0), size=length) + 1
        return (occur * sizes).astype(np.float64)

    # lumpy: rare events but large, variable sizes
    p = rng.uniform(0.05, 0.15)
    occur = rng.random(length) < p
    sizes = rng.gamma(shape=2.0, scale=rng.uniform(4, 10), size=length)
    return np.round(occur * sizes).astype(np.float64)


def make_synthetic_panel(
    n_series: int = 500,
    length: int = 730,
    seasonality: int = 7,
    seed: int = 42,
) -> SeriesPanel:
    """Generate ``n_series`` daily demand series of the given ``length``."""
    rng = np.random.default_rng(seed)
    start_date = np.datetime64("2014-01-01")

    series: list[TimeSeries] = []
    # Skew the mix toward the intermittent tail, as in real retail catalogues.
    weights = np.array([0.30, 0.20, 0.30, 0.20])
    for i in range(n_series):
        profile = _PROFILES[rng.choice(len(_PROFILES), p=weights)]
        values = _make_series(i, length, profile, rng)
        static = {
            "item_id": f"ITEM_{i:05d}",
            "cat_id": _CATEGORIES[i % len(_CATEGORIES)],
            "store_id": _STORES[i % len(_STORES)],
            "true_profile": profile,
        }
        series.append(
            TimeSeries(
                item_id=f"SYN_{i:05d}",
                values=values,
                start_date=start_date,
                freq="D",
                static=static,
            )
        )

    return SeriesPanel(series, name="synthetic", seasonality=seasonality)
