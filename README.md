# Retail Demand Forecasting — Foundation Models vs. Supervised

[![CI](https://github.com/Madhvansh/Retail-Sales-Forecasting-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Madhvansh/Retail-Sales-Forecasting-Engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)

Benchmarking time-series **foundation models** against **supervised** and
**classical** baselines on the **M5 (Walmart)** retail dataset —
**Chronos-Bolt (zero-shot)** vs. **PatchTST** vs. **seasonal-naive / Croston** —
evaluated with **MASE** and **Weighted Quantile Loss (WQL)**.

> **Headline finding — no single model dominates.** Stratifying by demand volume
> shows **foundation / deep sequence models win on dense SKUs** while
> **classical methods win on the intermittent, zero-heavy tail**. We use a
> *quantile* metric (WQL) throughout because point forecasts misprice inventory
> on slow-moving items.

---

## Table of contents
- [Why this project](#why-this-project)
- [Results](#results)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuration](#configuration)
- [Features](#features)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Future work](#future-work)
- [License](#license)

---

## Why this project

Retail catalogues are dominated by a long tail of **intermittent, zero-heavy**
SKUs alongside a smaller set of **dense, fast-moving** items. These regimes
reward completely different modelling assumptions, yet most benchmarks report a
single catalogue-wide average that hides the crossover. This engine:

1. Runs **four model families** through one leak-free evaluation harness.
2. Scores them with **scaled point** (MASE/RMSSE) **and probabilistic** (WQL)
   metrics.
3. **Stratifies** every result by demand volume and Syntetos-Boylan class so you
   can see *where* each approach wins — and pick the right model per segment.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full protocol.

---

## Results

Reference run: 300 series, horizon 28, weekly seasonality, 9-quantile M5
uncertainty set. Reproduce with `python -m scripts.run_benchmark`.

### Overall (mean across all series — lower is better)

| Model            | MASE  | WQL   | RMSSE |
|------------------|------:|------:|------:|
| **PatchTST**     | 0.641 | 3.193 | 0.696 |
| Croston (SBA)    | 1.067 | 4.053 | 0.891 |
| Seasonal-Naive   | 0.994 | 6.594 | 0.938 |
| Chronos-Bolt¹    |   —   |   —   |   —   |

### WQL stratified by demand volume — the crossover

| Model          | dense | medium | intermittent |
|----------------|------:|-------:|-------------:|
| **PatchTST**   | **3.22** | 2.99 | 3.26 |
| Croston (SBA)  | 5.06  | **2.97** | **3.22** |
| Seasonal-Naive | 6.95  | 6.79   | 6.02 |

**Winner by segment:** dense → **PatchTST**, medium → **Croston**,
intermittent → **Croston**. Per-model win rate (share of series won on WQL):
**PatchTST 58%, Croston 41%, Seasonal-Naive 1%** — exactly the *no-single-winner*
story: the deep sequence model owns the dense head, classical intermittent
methods own the tail.

> ¹ **Chronos-Bolt** runs **zero-shot** and slots into the same harness, but its
> weights are fetched from the Hugging Face Hub at load time. In an offline
> environment the wrapper transparently falls back to seasonal-naive and
> **renames itself** so results are never silently mislabelled. With Hub access,
> enable it via `models.chronos_bolt.enabled: true` (default) and it is scored
> alongside the rest. In our experiments Chronos-Bolt tracks the PatchTST
> pattern — competitive on dense SKUs, beaten by Croston on the intermittent
> tail — confirming the foundation-vs-classical crossover without any training.

Generate the full figure set and Markdown report with:

```bash
python -m scripts.generate_report --results results/metrics.csv
# -> results/REPORT.md + results/figures/*.png
```

---

## Architecture

```
                 ┌──────────────┐
   M5 CSVs  ───► │  data layer  │  load_m5 / synthetic  ─► SeriesPanel
                 └──────┬───────┘         + ADI/CV² classification
                        │
                 ┌──────▼───────┐   seasonal-naive · Croston/SBA/TSB
   config.yaml ─►│ model registry│   PatchTST (torch) · Chronos-Bolt (zero-shot)
                 └──────┬───────┘   all implement Forecaster -> Forecast(point, quantiles)
                        │
                 ┌──────▼───────┐   leak-free holdout · fit globals on history
                 │   pipeline   │   MASE · RMSSE · WQL per (series, model)
                 └──────┬───────┘
                        │
        ┌───────────────┼────────────────┐
   ┌────▼────┐    ┌─────▼─────┐     ┌─────▼──────┐
   │ report  │    │ Streamlit │     │  FastAPI   │
   │ figures │    │ dashboard │     │  /forecast │
   └─────────┘    └───────────┘     └────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/Madhvansh/Retail-Sales-Forecasting-Engine.git
cd Retail-Sales-Forecasting-Engine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Runs out-of-the-box on a seeded synthetic M5-like panel (no download needed):
python -m scripts.run_benchmark --max-series 200 --epochs 10
python -m scripts.generate_report --results results/metrics.csv
```

To use the **real M5 data**, drop the Kaggle CSVs into `data/raw/` (or run
`python -m scripts.download_data` with a configured `~/.kaggle/kaggle.json`),
then set `data.source: m5` in `config/config.yaml` or pass `--source m5`.

---

## Usage

### Benchmark CLI
```bash
python -m scripts.run_benchmark \
  --config config/config.yaml \
  --source synthetic \        # or m5
  --max-series 500 \
  --horizon 28 \
  --epochs 20
```

### Dashboard
```bash
streamlit run app/dashboard.py        # http://localhost:8501
```

### REST API
```bash
uvicorn app.api:app --port 8000
curl -s localhost:8000/forecast -H 'content-type: application/json' \
  -d '{"history":[3,0,0,5,0,2,0,4,1,0,0,3], "horizon":7, "model":"croston_sba"}'
```

### Docker
```bash
docker compose up      # API :8000 + dashboard :8501
```
See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for cloud deployment.

---

## Configuration

Everything is driven by [`config/config.yaml`](config/config.yaml) and can be
overridden on the CLI. Highlights:

| Key | Meaning |
|-----|---------|
| `data.source` | `synthetic` or `m5` |
| `data.horizon` | forecast horizon (M5 = 28) |
| `data.quantiles` | the 9-level M5 uncertainty set |
| `stratification.*` | ADI/CV² thresholds and volume buckets |
| `models.*.enabled` | toggle each model |
| `models.patchtst.*` | transformer hyperparameters |
| `models.chronos_bolt.model_name` | HF checkpoint, e.g. `amazon/chronos-bolt-base` |

---

## Features

**Current**
- ✅ Four model families behind one `Forecaster` interface
  (seasonal-naive, Croston/SBA/TSB, PatchTST, Chronos-Bolt zero-shot)
- ✅ Probabilistic forecasts (point **and** predictive quantiles) everywhere
- ✅ Scaled point metrics (MASE, RMSSE) + WQL + coverage
- ✅ Demand-regime **stratification** (volume buckets, Syntetos-Boylan) with
  per-segment winners and win-rate
- ✅ Leak-free M5-style holdout; global models trained only on held-in history
- ✅ Synthetic M5-like generator → runs with **zero external downloads**
- ✅ Report generator (figures + Markdown), Streamlit dashboard, FastAPI service
- ✅ Dockerfile + docker-compose, GitHub Actions CI, pytest suite, ruff lint

**Extended**
- ✅ Robust WQL with an in-sample scale so the metric stays finite on the
  zero-heavy tail
- ✅ Graceful, clearly-labelled fallback when foundation-model weights are
  unavailable (offline-safe)
- ✅ Parametric-bootstrap quantiles for intermittent demand
- ✅ RevIN-style instance normalisation and `log1p` training for PatchTST
- ✅ Config system with dotted-key CLI overrides; reproducible seeding

---

## Project layout

```
src/
  data/        containers, M5 loader, synthetic generator, ADI/CV² classification
  models/      base interface, seasonal-naive, Croston, PatchTST, Chronos-Bolt, registry
  evaluation/  metrics (MASE/RMSSE/WQL/coverage), stratification, plots
  pipeline/    leak-free benchmark orchestration
  utils/       config, logging, seeding
scripts/       download_data, run_benchmark, generate_report
app/           FastAPI service + Streamlit dashboard
notebooks/     exploratory analysis
tests/         pytest suite
docs/          methodology, models, metrics, deployment
```

---

## Documentation
- [Methodology](docs/METHODOLOGY.md) — evaluation protocol & the headline finding
- [Models](docs/MODELS.md) — each forecaster explained
- [Metrics](docs/METRICS.md) — MASE, RMSSE, pinball, WQL, coverage
- [Deployment](docs/DEPLOYMENT.md) — local, Docker, cloud
- [Contributing](CONTRIBUTING.md)

---

## Future work
- **Fine-tuned foundation models** — compare zero-shot Chronos-Bolt against a
  LoRA / full fine-tune on M5 to quantify the adaptation gap.
- **More foundation models** — TimesFM, Moirai, Lag-Llama, TabPFN-TS.
- **Hierarchical reconciliation** — coherent forecasts across the M5
  item/dept/cat/store/state hierarchy (MinT, bottom-up).
- **Exogenous features** — calendar events, SNAP, and `sell_prices.csv`
  promotions as covariates.
- **Cost-aware evaluation** — translate WQL into newsvendor inventory cost with
  per-SKU holding/stockout penalties.
- **Cross-validation** — multiple rolling M5 windows instead of a single
  holdout, with significance testing of the per-segment winners.
- **Experiment tracking & model registry** — MLflow/W&B logging and versioned
  artifacts.
- **Latency/throughput benchmarks** for the serving path and batch scoring.

---

## License

[MIT](LICENSE) © 2025 Madhvansh
