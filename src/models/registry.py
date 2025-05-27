"""Build the set of enabled models from a config block."""

from __future__ import annotations

import numpy as np

from src.models.base import Forecaster
from src.models.croston import Croston
from src.models.seasonal_naive import SeasonalNaive
from src.utils import Config, get_logger

log = get_logger("models.registry")


def build_models(cfg: Config, quantile_levels: np.ndarray, seed: int = 42) -> list[Forecaster]:
    """Instantiate every model whose ``enabled`` flag is true."""
    horizon = cfg.data.horizon
    context = cfg.data.context_length
    season = cfg.data.get("seasonality", 7)
    m = cfg.models

    models: list[Forecaster] = []

    if m.get("seasonal_naive", {}).get("enabled", False):
        models.append(SeasonalNaive(season_length=m.seasonal_naive.get("season_length", season)))

    if m.get("croston", {}).get("enabled", False):
        c = m.croston
        models.append(Croston(variant=c.get("variant", "sba"), alpha=c.get("alpha", 0.1), seed=seed))

    if m.get("patchtst", {}).get("enabled", False):
        from src.models.patchtst import PatchTST

        p = m.patchtst
        models.append(
            PatchTST(
                context_length=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                patch_len=p.get("patch_len", 16),
                stride=p.get("stride", 8),
                d_model=p.get("d_model", 128),
                n_heads=p.get("n_heads", 8),
                n_layers=p.get("n_layers", 3),
                dropout=p.get("dropout", 0.2),
                epochs=p.get("epochs", 20),
                batch_size=p.get("batch_size", 64),
                lr=p.get("lr", 1e-3),
                seed=seed,
            )
        )

    if m.get("chronos_bolt", {}).get("enabled", False):
        from src.models.chronos_bolt import ChronosBolt

        cb = m.chronos_bolt
        models.append(
            ChronosBolt(
                model_name=cb.get("model_name", "amazon/chronos-bolt-small"),
                device=cb.get("device", "auto"),
                quantile_levels=quantile_levels,
                season_length=season,
            )
        )

    log.info("Built %d models: %s", len(models), ", ".join(mdl.name for mdl in models))
    return models
