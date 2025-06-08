"""Run the full forecasting benchmark from the command line.

Examples
--------
# Quick smoke run on synthetic data:
python -m scripts.run_benchmark --max-series 50 --epochs 3

# Full M5 run (after `python -m scripts.download_data`):
python -m scripts.run_benchmark --source m5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_panel
from src.evaluation.stratify import overall_table, stratified_table, win_rate, winners
from src.models import build_models
from src.pipeline import run_benchmark
from src.utils import get_logger, load_config, set_seed

log = get_logger("run_benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail forecasting benchmark.")
    p.add_argument("--config", default="config/config.yaml", type=Path)
    p.add_argument("--source", choices=["synthetic", "m5"], default=None)
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override PatchTST epochs")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    overrides: dict = {}
    if args.source is not None:
        overrides["data.source"] = args.source
    if args.max_series is not None:
        overrides["data.max_series"] = args.max_series
    if args.horizon is not None:
        overrides["data.horizon"] = args.horizon
    if args.epochs is not None:
        overrides["models.patchtst.epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = load_config(args.config, overrides)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    out_dir = Path(args.out_dir or cfg.evaluation.get("report_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    quantile_levels = np.asarray(cfg.data.quantiles, dtype=float)

    panel = build_panel(cfg, seed=seed)
    models = build_models(cfg, quantile_levels, seed=seed)
    results = run_benchmark(panel, models, cfg)

    metrics_path = out_dir / "metrics.csv"
    results.to_csv(metrics_path, index=False)
    log.info("Wrote per-series metrics -> %s", metrics_path)

    overall = overall_table(results)
    log.info("\n=== Overall (mean across all series) ===\n%s", overall.to_string())

    for by in ("volume_bucket", "sb_class"):
        tbl = stratified_table(results, by=by, metric="wql")
        log.info("\n=== WQL by %s ===\n%s", by, tbl.round(4).to_string())
        win = winners(results, by=by, metric="wql")
        log.info("\n=== Winner by %s ===\n%s", by, win.to_string(index=False))

    wr = win_rate(results, metric="wql")
    log.info("\n=== Per-model win rate (WQL) ===\n%s", wr.to_string(index=False))

    overall.to_csv(out_dir / "summary_overall.csv")
    winners(results, by="volume_bucket").to_csv(out_dir / "summary_winners_volume.csv", index=False)
    log.info("Done. Summaries in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
