"""Core data containers shared across loaders, models and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeSeries:
    """A single univariate demand series with optional static metadata."""

    item_id: str
    values: np.ndarray  # shape (T,), float
    start_date: np.datetime64 | None = None
    freq: str = "D"
    static: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64).ravel()

    def __len__(self) -> int:
        return self.values.shape[0]

    def split(self, horizon: int) -> tuple["TimeSeries", np.ndarray]:
        """Return (history, future) where future has length ``horizon``."""
        if horizon >= len(self):
            raise ValueError(
                f"horizon {horizon} >= series length {len(self)} for {self.item_id}"
            )
        hist = TimeSeries(
            item_id=self.item_id,
            values=self.values[:-horizon],
            start_date=self.start_date,
            freq=self.freq,
            static=dict(self.static),
        )
        future = self.values[-horizon:]
        return hist, future


@dataclass
class SeriesPanel:
    """A collection of :class:`TimeSeries` plus dataset-level metadata."""

    series: list[TimeSeries]
    name: str = "panel"
    seasonality: int = 7

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self):
        return iter(self.series)

    def __getitem__(self, idx: int) -> TimeSeries:
        return self.series[idx]

    def subset(self, n: int | None) -> "SeriesPanel":
        if n is None or n >= len(self.series):
            return self
        return SeriesPanel(self.series[:n], name=self.name, seasonality=self.seasonality)

    @property
    def item_ids(self) -> list[str]:
        return [s.item_id for s in self.series]

    def matrix(self) -> np.ndarray:
        """Stack series into a (N, T) array (requires equal length)."""
        lengths = {len(s) for s in self.series}
        if len(lengths) != 1:
            raise ValueError("matrix() requires all series to share one length")
        return np.vstack([s.values for s in self.series])
