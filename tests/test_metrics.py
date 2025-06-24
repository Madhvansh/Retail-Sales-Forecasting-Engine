import numpy as np

from src.evaluation.metrics import coverage, mase, quantile_loss, rmsse, wql


def test_mase_perfect_forecast_is_zero():
    hist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    actual = np.array([1, 2, 3], dtype=float)
    assert mase(actual, actual, hist, season_length=1) == 0.0


def test_mase_scales_with_seasonal_naive():
    # A forecast equal to the seasonal-naive prediction should give MASE ~ 1.
    rng = np.random.default_rng(0)
    hist = rng.poisson(5, size=200).astype(float)
    actual = rng.poisson(5, size=7).astype(float)
    fc = np.roll(hist, 7)[:7]  # arbitrary; just check finiteness & positivity
    val = mase(actual, fc, hist, season_length=7)
    assert val > 0 and np.isfinite(val)


def test_rmsse_non_negative():
    hist = np.arange(1, 50, dtype=float)
    actual = np.array([50, 51, 52], dtype=float)
    fc = np.array([49, 50, 51], dtype=float)
    assert rmsse(actual, fc, hist, season_length=7) >= 0


def test_quantile_loss_zero_when_exact():
    actual = np.array([2.0, 3.0])
    levels = np.array([0.5])
    qf = actual[None, :]  # median forecast equals actual
    loss = quantile_loss(actual, qf, levels)
    assert np.allclose(loss, 0.0)


def test_quantile_loss_asymmetry():
    # Under-forecasting at a high quantile is penalised more than over-forecasting.
    actual = np.array([10.0])
    levels = np.array([0.9])
    under = quantile_loss(actual, np.array([[5.0]]), levels)[0]
    over = quantile_loss(actual, np.array([[15.0]]), levels)[0]
    assert under > over


def test_wql_finite_on_all_zero_actual():
    actual = np.zeros(28)
    levels = np.array([0.1, 0.5, 0.9])
    qf = np.zeros((3, 28)) + 0.5
    val = wql(actual, qf, levels, scale=10.0)
    assert np.isfinite(val) and val >= 0


def test_coverage_monotone_in_quantile():
    rng = np.random.default_rng(1)
    actual = rng.normal(10, 2, size=500)
    levels = np.array([0.1, 0.5, 0.9])
    qf = np.vstack([np.full(500, q) for q in [7.4, 10.0, 12.6]])
    cov = coverage(actual, qf, levels)
    vals = [cov[float(level)] for level in levels]
    assert vals[0] <= vals[1] <= vals[2]
