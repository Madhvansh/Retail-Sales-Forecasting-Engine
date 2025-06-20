"""Forecasting service used by the API: lazy model construction + inference."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.data.dataset import TimeSeries
from src.models.base import Forecaster
from src.models.croston import Croston
from src.models.seasonal_naive import SeasonalNaive
from src.utils import get_logger

log = get_logger("service")

DEFAULT_QUANTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


@lru_cache(maxsize=8)
def _get_model(name: str, season_length: int = 7) -> Forecaster:
    """Construct (and cache) a model by name. Heavy models load lazily."""
    name = name.lower()
    if name in {"seasonal_naive", "naive"}:
        return SeasonalNaive(season_length=season_length)
    if name in {"croston", "croston_sba", "sba"}:
        return Croston(variant="sba")
    if name in {"croston_classic", "classic"}:
        return Croston(variant="classic")
    if name in {"croston_tsb", "tsb"}:
        return Croston(variant="tsb")
    if name in {"chronos", "chronos_bolt"}:
        from src.models.chronos_bolt import ChronosBolt

        return ChronosBolt(season_length=season_length)
    raise ValueError(f"Unknown model '{name}'")


def available_models() -> list[str]:
    models = ["seasonal_naive", "croston_sba", "croston_classic", "croston_tsb"]
    try:  # only advertise chronos if it can be imported
        import chronos  # noqa: F401

        models.append("chronos_bolt")
    except ImportError:
        pass
    return models


def forecast(
    history: list[float],
    horizon: int,
    model: str = "croston_sba",
    quantile_levels: list[float] | None = None,
    season_length: int = 7,
) -> dict:
    """Return point + quantile forecast for a single history series."""
    levels = np.asarray(quantile_levels or DEFAULT_QUANTILES, dtype=float)
    ts = TimeSeries(item_id="request", values=np.asarray(history, dtype=float))
    mdl = _get_model(model, season_length=season_length)
    fc = mdl.predict(ts, horizon, levels)
    return {
        "model": mdl.name,
        "horizon": horizon,
        "point": fc.point.tolist(),
        "quantile_levels": fc.quantile_levels.tolist(),
        "quantiles": {
            str(round(float(q), 4)): fc.quantiles[i].tolist()
            for i, q in enumerate(fc.quantile_levels)
        },
    }
