# Models

All models implement the common `Forecaster` interface in
`src/models/base.py` and emit a `Forecast` carrying a point forecast plus a
`(Q, H)` matrix of predictive quantiles.

| Model | Family | Trained on M5? | Probabilistic via |
|-------|--------|----------------|-------------------|
| Seasonal-Naive | classical | no (local) | seasonal residual quantiles |
| Croston / SBA / TSB | classical (intermittent) | no (local) | parametric bootstrap |
| PatchTST | supervised deep | **yes** (global) | multi-quantile pinball head |
| Chronos-Bolt | foundation | **no — zero-shot** | native quantile head |

## Seasonal-Naive (`seasonal_naive.py`)
Repeats the last seasonal cycle (period 7 for daily retail). Predictive
quantiles come from in-sample seasonal residuals, scaled by √h. A deceptively
strong probabilistic baseline on dense, seasonal series.

## Croston family (`croston.py`)
The standard toolkit for **intermittent demand**. Separately smooths demand
*sizes* and *intervals*:
- **classic** — `z / p`
- **SBA** (Syntetos-Boylan Approximation) — bias-corrected `(1 − α/2)·z/p`
- **TSB** (Teunter-Syntetos-Babai) — probability-based, robust to obsolescence

Quantiles come from a parametric bootstrap (Bernoulli occurrence × resampled
non-zero sizes), which respects the zero-inflated distribution that WQL rewards.

## PatchTST (`patchtst.py`)
A channel-independent **patched Transformer** (Nie et al., ICLR 2023),
implemented from scratch in PyTorch. Trained globally across the panel in
`log1p` space with RevIN-style per-window instance normalisation and a
**multi-quantile pinball** loss head, so it is directly comparable to the
foundation model on the probabilistic metric.

Key knobs (`config.yaml → models.patchtst`): `patch_len`, `stride`, `d_model`,
`n_heads`, `n_layers`, `epochs`.

## Chronos-Bolt (`chronos_bolt.py`)
Amazon's **T5-based time-series foundation model**, used **zero-shot** (no
fine-tuning on M5). We read predictive quantiles directly from
`predict_quantiles`. If the package/weights are unavailable, the wrapper falls
back to seasonal-naive and renames itself loudly so results are never silently
mislabelled. Configure the checkpoint via `models.chronos_bolt.model_name`
(e.g. `amazon/chronos-bolt-small` / `-base`).

## Adding a model
1. Subclass `Forecaster`, implement `predict()` (and `fit()` if global).
2. Register it in `src/models/registry.py` behind a config flag.
3. Add a unit test in `tests/test_models.py`.
