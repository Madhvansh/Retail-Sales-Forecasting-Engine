"""Data loading: M5 panel, synthetic generator and shared containers."""

from src.data.build import build_panel
from src.data.dataset import SeriesPanel, TimeSeries
from src.data.m5 import load_m5
from src.data.preprocessing import DemandStats, classify_panel, classify_series
from src.data.synthetic import make_synthetic_panel

__all__ = [
    "SeriesPanel",
    "TimeSeries",
    "load_m5",
    "make_synthetic_panel",
    "build_panel",
    "DemandStats",
    "classify_series",
    "classify_panel",
]
