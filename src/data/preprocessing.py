"""Per-series demand statistics and Syntetos-Boylan classification.

These features drive the central analysis of the project: stratifying error
metrics by demand regime so we can see *where* each model wins rather than
relying on a single dataset-wide average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import SeriesPanel, TimeSeries


@dataclass
class DemandStats:
    item_id: str
    mean: float
    adi: float          # average inter-demand interval
    cv2: float          # squared coefficient of variation of non-zero demand
    zero_frac: float
    sb_class: str       # smooth | erratic | intermittent | lumpy
    volume_bucket: str  # intermittent | medium | dense
    extra: dict = field(default_factory=dict)


def _adi(values: np.ndarray) -> float:
    """Average interval between non-zero demand periods."""
    nz = np.flatnonzero(values > 0)
    if nz.size <= 1:
        return float(len(values)) if nz.size else np.inf
    return float(len(values) / nz.size)


def _cv2(values: np.ndarray) -> float:
    """Squared coefficient of variation of the non-zero demand sizes."""
    nz = values[values > 0]
    if nz.size < 2 or nz.mean() == 0:
        return 0.0
    return float((nz.std(ddof=0) / nz.mean()) ** 2)


def classify_series(
    ts: TimeSeries,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    volume_buckets: tuple[float, ...] = (0.0, 1.0, 5.0, 1e9),
    volume_labels: tuple[str, ...] = ("intermittent", "medium", "dense"),
) -> DemandStats:
    """Compute demand statistics and assign regime labels for one series."""
    v = ts.values
    adi = _adi(v)
    cv2 = _cv2(v)
    mean = float(v.mean())
    zero_frac = float((v == 0).mean())

    # Syntetos-Boylan (2005) classification quadrants.
    if adi < adi_threshold and cv2 < cv2_threshold:
        sb = "smooth"
    elif adi >= adi_threshold and cv2 < cv2_threshold:
        sb = "intermittent"
    elif adi < adi_threshold and cv2 >= cv2_threshold:
        sb = "erratic"
    else:
        sb = "lumpy"

    bucket = volume_labels[-1]
    for i in range(len(volume_labels)):
        if volume_buckets[i] <= mean < volume_buckets[i + 1]:
            bucket = volume_labels[i]
            break

    return DemandStats(
        item_id=ts.item_id,
        mean=mean,
        adi=adi,
        cv2=cv2,
        zero_frac=zero_frac,
        sb_class=sb,
        volume_bucket=bucket,
        extra={"true_profile": ts.static.get("true_profile")},
    )


def classify_panel(panel: SeriesPanel, **kwargs) -> dict[str, DemandStats]:
    """Classify every series in a panel, keyed by ``item_id``."""
    return {ts.item_id: classify_series(ts, **kwargs) for ts in panel}
