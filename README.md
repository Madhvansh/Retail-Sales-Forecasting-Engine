# Retail Demand Forecasting — Foundation Models vs. Supervised

Benchmarking time-series **foundation models** against **supervised** and
**classical** baselines on the M5 (Walmart) retail dataset.

> Work in progress. See `docs/` for design notes.

## Goal

Evaluate whether zero-shot time-series foundation models can replace
purpose-built forecasters for retail demand planning.

## Models under test

- **Chronos-Bolt** (zero-shot foundation model)
- **PatchTST** (supervised transformer)
- **Seasonal-Naive** / **Croston** (classical baselines)

## Metrics

- **MASE** — Mean Absolute Scaled Error
- **WQL** — Weighted Quantile Loss

## License

MIT
