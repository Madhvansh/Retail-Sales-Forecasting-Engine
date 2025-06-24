import numpy as np

from src.data.dataset import SeriesPanel, TimeSeries
from src.data.preprocessing import classify_series
from src.data.synthetic import make_synthetic_panel


def test_synthetic_panel_shape():
    panel = make_synthetic_panel(n_series=20, length=400, seed=0)
    assert isinstance(panel, SeriesPanel)
    assert len(panel) == 20
    assert all(len(ts) == 400 for ts in panel)


def test_synthetic_is_reproducible():
    a = make_synthetic_panel(n_series=10, length=200, seed=7).matrix()
    b = make_synthetic_panel(n_series=10, length=200, seed=7).matrix()
    np.testing.assert_array_equal(a, b)


def test_timeseries_split():
    ts = TimeSeries(item_id="x", values=np.arange(100, dtype=float))
    hist, future = ts.split(horizon=10)
    assert len(hist) == 90
    assert len(future) == 10
    np.testing.assert_array_equal(future, np.arange(90, 100))


def test_classify_intermittent_vs_smooth():
    # Smooth/dense series.
    dense = TimeSeries(item_id="d", values=np.full(200, 10.0) + np.random.default_rng(0).normal(0, 1, 200))
    stats_dense = classify_series(dense)
    assert stats_dense.volume_bucket == "dense"
    assert stats_dense.zero_frac == 0.0

    # Intermittent series (lots of zeros).
    inter_vals = np.zeros(200)
    inter_vals[::8] = 1.0
    stats_inter = classify_series(TimeSeries(item_id="i", values=inter_vals))
    assert stats_inter.adi > 1.32
    assert stats_inter.zero_frac > 0.5
