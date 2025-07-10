# Methodology

## Question

> Can a zero-shot time-series **foundation model** (Chronos-Bolt) replace
> purpose-built supervised and classical forecasters for retail demand
> planning — and if not, *where* does each approach win?

We answer this by benchmarking on the **M5 (Walmart)** dataset and, crucially,
by **stratifying** the error by demand regime instead of reporting a single
catalogue-wide average.

## Evaluation protocol

- **Holdout**: the last `horizon` (default 28) observations of every series are
  held out, mirroring the M5 validation window.
- **No leakage**: global supervised models (PatchTST) are trained *only* on the
  held-in history; classical and zero-shot models see only the history at
  forecast time.
- **Probabilistic**: every model emits both a point forecast and a set of
  predictive quantiles (the M5 uncertainty set), because inventory decisions
  are driven by service-level quantiles, not the mean.

## Why stratify?

A dataset-wide average hides the real story. Retail catalogues are dominated by
a long **intermittent, zero-heavy tail** of slow-moving SKUs alongside a
smaller set of **dense, fast-moving** items. These regimes reward completely
different modelling assumptions, so we split results two ways:

1. **Demand volume** — mean daily units bucketed into
   `intermittent / medium / dense`.
2. **Syntetos-Boylan class** — `smooth / erratic / intermittent / lumpy`,
   derived from the average inter-demand interval (ADI) and the squared
   coefficient of variation of non-zero demand (CV²):

   | | CV² < 0.49 | CV² ≥ 0.49 |
   |---|---|---|
   | **ADI < 1.32** | smooth | erratic |
   | **ADI ≥ 1.32** | intermittent | lumpy |

## Headline finding

**No single model dominates.** Foundation/supervised sequence models win on
**dense** SKUs, where there is enough signal to learn seasonality and trend,
while **classical intermittent-demand methods (Croston/SBA)** win on the
**intermittent, zero-heavy tail**, where deep models over-forecast and waste
inventory. Reporting only the average would have masked this crossover — and
chosen the wrong model for the majority (tail) of the catalogue.

## Reproducing

```bash
python -m scripts.run_benchmark --config config/config.yaml
python -m scripts.generate_report --results results/metrics.csv
```

Results are deterministic given `seed` (default 42); the synthetic generator is
seeded so the pipeline is runnable and reproducible without the Kaggle download.
