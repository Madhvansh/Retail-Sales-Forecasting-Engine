import numpy as np
import pytest

from src.data.dataset import TimeSeries
from src.models.croston import Croston
from src.models.seasonal_naive import SeasonalNaive

QUANTILES = np.array([0.1, 0.5, 0.9])


def _ts(values):
    return TimeSeries(item_id="t", values=np.asarray(values, dtype=float))


def test_seasonal_naive_repeats_last_season():
    ts = _ts(list(range(1, 22)))  # 21 points, season 7
    fc = SeasonalNaive(season_length=7).predict(ts, horizon=7, quantile_levels=QUANTILES)
    assert fc.point.shape == (7,)
    np.testing.assert_allclose(fc.point, np.array([15, 16, 17, 18, 19, 20, 21], dtype=float))


def test_forecast_quantiles_are_monotone():
    rng = np.random.default_rng(0)
    ts = _ts(rng.poisson(5, size=120))
    fc = SeasonalNaive(season_length=7).predict(ts, horizon=14, quantile_levels=QUANTILES)
    # q10 <= q50 <= q90 at every horizon step
    assert np.all(fc.quantiles[0] <= fc.quantiles[1] + 1e-8)
    assert np.all(fc.quantiles[1] <= fc.quantiles[2] + 1e-8)


def test_croston_handles_all_zero_history():
    ts = _ts(np.zeros(60))
    fc = Croston(variant="sba").predict(ts, horizon=7, quantile_levels=QUANTILES)
    assert np.all(fc.point == 0.0)
    assert fc.quantiles.shape == (3, 7)


def test_croston_intermittent_point_is_positive():
    values = np.zeros(100)
    values[::5] = 3  # demand every 5 periods
    fc = Croston(variant="classic").predict(_ts(values), horizon=10, quantile_levels=QUANTILES)
    assert fc.point[0] > 0


@pytest.mark.parametrize("variant", ["classic", "sba", "tsb"])
def test_croston_variants_run(variant):
    rng = np.random.default_rng(2)
    values = (rng.random(80) < 0.2) * rng.poisson(2, 80)
    fc = Croston(variant=variant).predict(_ts(values), horizon=7, quantile_levels=QUANTILES)
    assert np.all(np.isfinite(fc.point))
    assert np.all(fc.quantiles >= 0)
