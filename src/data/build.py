"""Build the working panel from a config (M5 or synthetic)."""

from __future__ import annotations

from src.data.dataset import SeriesPanel
from src.data.m5 import load_m5
from src.data.synthetic import make_synthetic_panel
from src.utils import Config, get_logger

log = get_logger("data.build")


def build_panel(cfg: Config, seed: int = 42) -> SeriesPanel:
    """Return a :class:`SeriesPanel` according to ``cfg.data.source``."""
    source = cfg.data.get("source", "synthetic")
    max_series = cfg.data.get("max_series")
    seasonality = cfg.data.get("seasonality", 7)

    if source == "m5":
        panel = load_m5(cfg.data.raw_dir, max_series=max_series, seasonality=seasonality)
    elif source == "synthetic":
        panel = make_synthetic_panel(
            n_series=max_series or 500,
            length=cfg.data.get("context_length", 168) + cfg.data.get("horizon", 28) + 365,
            seasonality=seasonality,
            seed=seed,
        )
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown data.source: {source!r}")

    panel = panel.subset(max_series)
    log.info("Built panel '%s' with %d series", panel.name, len(panel))
    return panel
