import numpy as np

from src.data import build_panel
from src.evaluation.stratify import overall_table, stratified_table, win_rate, winners
from src.models import build_models
from src.pipeline import run_benchmark
from src.utils import load_config, set_seed


def _cfg():
    return load_config(
        "config/config.yaml",
        {
            "data.max_series": 24,
            "data.horizon": 14,
            "models.patchtst.enabled": False,
            "models.chronos_bolt.enabled": False,
        },
    )


def test_benchmark_runs_and_scores():
    cfg = _cfg()
    set_seed(0)
    ql = np.asarray(cfg.data.quantiles, dtype=float)
    panel = build_panel(cfg, seed=0)
    models = build_models(cfg, ql, seed=0)
    df = run_benchmark(panel, models, cfg)

    assert len(df) == len(panel) * len(models)
    for col in ("mase", "wql", "rmsse", "sb_class", "volume_bucket"):
        assert col in df.columns
    assert np.isfinite(df[["mase", "wql", "rmsse"]].to_numpy()).all()


def test_aggregations_are_consistent():
    cfg = _cfg()
    set_seed(1)
    ql = np.asarray(cfg.data.quantiles, dtype=float)
    panel = build_panel(cfg, seed=1)
    models = build_models(cfg, ql, seed=1)
    df = run_benchmark(panel, models, cfg)

    overall = overall_table(df)
    assert set(overall.index) == {m.name for m in models}

    pivot = stratified_table(df, by="volume_bucket", metric="wql")
    assert pivot.shape[0] == len(models)

    win = winners(df, by="volume_bucket", metric="wql")
    assert "winner" in win.columns

    rates = win_rate(df, metric="wql")
    assert abs(rates["win_rate"].sum() - 1.0) < 1e-6
