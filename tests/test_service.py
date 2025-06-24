import numpy as np

from app.service import available_models, forecast


def test_available_models_includes_classical():
    models = available_models()
    assert "seasonal_naive" in models
    assert "croston_sba" in models


def test_forecast_shape_and_keys():
    history = list(np.tile([3, 0, 0, 5, 0, 2, 0], 8).astype(float))
    out = forecast(history, horizon=7, model="croston_sba")
    assert out["horizon"] == 7
    assert len(out["point"]) == 7
    assert set(out.keys()) == {"model", "horizon", "point", "quantile_levels", "quantiles"}
    for series in out["quantiles"].values():
        assert len(series) == 7


def test_forecast_default_quantiles():
    history = [float(x) for x in range(1, 30)]
    out = forecast(history, horizon=5, model="seasonal_naive")
    assert out["quantile_levels"] == [0.1, 0.25, 0.5, 0.75, 0.9]
