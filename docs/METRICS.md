# Metrics

Implemented in `src/evaluation/metrics.py`. All point metrics are **scaled** so
they can be averaged across series with wildly different demand volumes.

## MASE — Mean Absolute Scaled Error
```
MASE = mean(|y − ŷ|) / mean(|y_t − y_{t−m}|)   # in-sample seasonal-naive MAE
```
Scale-free; `MASE < 1` beats the in-sample seasonal-naive benchmark, `> 1`
loses to it. `m` is the seasonal period (7).

## RMSSE — Root Mean Squared Scaled Error
The M5 *Accuracy* track metric: the RMSE analogue of MASE, using the squared
seasonal difference as the denominator.

## Pinball / Quantile loss
For quantile level `q` and forecast `ŷ_q`:
```
L_q = max(q·(y − ŷ_q), (q − 1)·(y − ŷ_q))
```
Asymmetric: at high `q`, under-forecasting is penalised more than
over-forecasting — the property that makes it correct for inventory.

## WQL — Weighted Quantile Loss
The total pinball loss across all quantiles and horizon steps, normalised by a
scale:
```
WQL = 2 · Σ_q Σ_h L_{q,h} / scale
```
- **Dataset convention** (default): `scale = Σ|y|` over the test window
  (GluonTS/Chronos definition).
- **Per-series stratified analysis** (what the pipeline uses): `scale =
  seasonal-naive MAE × horizon`, an *in-sample* denominator. This keeps WQL
  finite and comparable even when a slow-moving series has **zero demand in the
  test window** — the dataset convention divides by ≈0 and explodes on the
  intermittent tail.

**Why WQL is the headline metric.** Point forecasts misprice inventory: a
median of 0 for an intermittent SKU looks accurate on MAE/MASE yet provides no
basis for a safety stock. WQL scores the *whole predictive distribution*, which
is what a planner actually orders against.

## Coverage
Empirical fraction of actuals at or below each predicted quantile — a
calibration diagnostic (well-calibrated ⇒ coverage ≈ quantile level).
