"""Turn a ``metrics.csv`` into figures and a Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.plots import generate_all
from src.evaluation.stratify import overall_table, win_rate, winners
from src.utils import get_logger

log = get_logger("report")


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=index) + "\n```"


def build_report(results_path: Path, out_dir: Path, primary: str = "wql") -> Path:
    df = pd.read_csv(results_path)
    fig_dir = out_dir / "figures"
    paths = generate_all(df, fig_dir, primary=primary)
    log.info("Wrote %d figures to %s", len(paths), fig_dir)

    overall = overall_table(df)
    win_vol = winners(df, by="volume_bucket", metric=primary)
    win_sb = winners(df, by="sb_class", metric=primary)
    rates = win_rate(df, metric=primary)

    report = out_dir / "REPORT.md"
    lines = [
        "# Benchmark Report",
        "",
        f"Series evaluated: **{df['item_id'].nunique()}**  |  "
        f"Models: **{df['model'].nunique()}**  |  Primary metric: **{primary.upper()}**",
        "",
        "## Overall (mean across all series)",
        "",
        _md_table(overall.round(4)),
        "",
        "## Winner by demand volume",
        "",
        _md_table(win_vol, index=False),
        "",
        "## Winner by Syntetos-Boylan class",
        "",
        _md_table(win_sb, index=False),
        "",
        "## Per-model win rate",
        "",
        _md_table(rates, index=False),
        "",
        "## Figures",
        "",
        f"![overall](figures/overall_{primary}.png)",
        f"![by volume](figures/{primary}_by_volume.png)",
        f"![by class](figures/{primary}_by_sbclass.png)",
        "![win rate](figures/win_rate.png)",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote report -> %s", report)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate benchmark report and figures.")
    p.add_argument("--results", default="results/metrics.csv", type=Path)
    p.add_argument("--out-dir", default="results", type=Path)
    p.add_argument("--metric", default="wql")
    args = p.parse_args(argv)

    if not args.results.exists():
        log.error("Results file %s not found. Run the benchmark first.", args.results)
        return 1
    build_report(args.results, args.out_dir, primary=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
