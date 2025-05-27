"""Chronos-Bolt — zero-shot time-series foundation model.

Wraps Amazon's ``chronos-forecasting`` package. Chronos-Bolt is a T5-based
model pretrained on a large corpus of time series; we use it **zero-shot**
(no fine-tuning on M5) and read off predictive quantiles directly via
``predict_quantiles``.

If the package or weights are unavailable, the model can optionally fall back
to a seasonal-naive forecast so the pipeline still completes end-to-end; the
fallback is logged loudly so results are never silently mislabelled.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import TimeSeries
from src.models.base import Forecast, Forecaster
from src.models.seasonal_naive import SeasonalNaive
from src.utils import get_logger

log = get_logger("models.chronos")


class ChronosBolt(Forecaster):
    name = "chronos_bolt"
    is_global = False  # zero-shot: a single pretrained model, no panel training

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device: str = "auto",
        quantile_levels: np.ndarray | None = None,
        allow_fallback: bool = True,
        season_length: int = 7,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantile_levels = np.asarray(
            quantile_levels if quantile_levels is not None else [0.1, 0.5, 0.9],
            dtype=np.float64,
        )
        self.allow_fallback = allow_fallback
        self.season_length = season_length
        self._pipeline = None
        self._fallback = None
        self._load()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        try:
            from chronos import BaseChronosPipeline

            device = self._resolve_device()
            log.info("Loading %s on %s (zero-shot)", self.model_name, device)
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name, device_map=device
            )
        except Exception as exc:  # noqa: BLE001 - want a single broad guard here
            if not self.allow_fallback:
                raise
            log.warning(
                "Could not load Chronos-Bolt (%s). Falling back to seasonal-naive. "
                "Install `chronos-forecasting` and ensure network access for the real model.",
                exc,
            )
            self._fallback = SeasonalNaive(season_length=self.season_length)
            self.name = "chronos_bolt[fallback=seasonal_naive]"

    def predict(
        self,
        history: TimeSeries,
        horizon: int,
        quantile_levels: np.ndarray,
    ) -> Forecast:
        if self._fallback is not None:
            return self._fallback.predict(history, horizon, quantile_levels)

        import torch

        context = torch.tensor(history.values, dtype=torch.float32)
        levels = list(np.asarray(quantile_levels, dtype=float))
        # predict_quantiles returns (quantiles[B, H, Q], mean[B, H]).
        q, mean = self._pipeline.predict_quantiles(
            context=context,
            prediction_length=horizon,
            quantile_levels=levels,
        )
        quantiles = q[0].cpu().numpy().T          # (Q, H)
        quantiles = np.clip(quantiles, 0.0, None)
        point = np.clip(mean[0].cpu().numpy(), 0.0, None)

        return Forecast(
            item_id=history.item_id,
            point=point,
            quantiles=quantiles,
            quantile_levels=np.asarray(quantile_levels),
        )
