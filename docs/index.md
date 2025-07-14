# Retail Demand Forecasting — Foundation Models vs. Supervised

Benchmarking time-series **foundation models** against **supervised** and
**classical** baselines on the **M5 (Walmart)** retail dataset —
**Chronos-Bolt (zero-shot)** vs. **PatchTST** vs. **seasonal-naive / Croston** —
evaluated with **MASE** and **Weighted Quantile Loss (WQL)**.

> **Headline finding — no single model dominates.** Stratifying by demand volume
> shows **foundation / deep sequence models win on dense SKUs** while
> **classical methods win on the intermittent, zero-heavy tail**. We use a
> *quantile* metric (WQL) throughout, because point forecasts misprice inventory
> on slow-moving items.

[View the code on GitHub »](https://github.com/Madhvansh/Retail-Sales-Forecasting-Engine)

---

## Results at a glance

WQL stratified by demand volume — the crossover (lower is better):

| Model          | dense | medium | intermittent |
|----------------|------:|-------:|-------------:|
| **PatchTST**   | **3.22** | 2.99 | 3.26 |
| Croston (SBA)  | 5.06  | **2.97** | **3.22** |
| Seasonal-Naive | 6.95  | 6.79   | 6.02 |

**Winner by segment:** dense → PatchTST, medium → Croston, intermittent → Croston.
Per-model win rate: **PatchTST 58%, Croston 41%, Seasonal-Naive 1%**.

---

## Documentation

- [Methodology](METHODOLOGY.html) — evaluation protocol & the headline finding
- [Models](MODELS.html) — each forecaster explained
- [Metrics](METRICS.html) — MASE, RMSSE, pinball, WQL, coverage
- [Deployment](DEPLOYMENT.html) — local, Docker, cloud

---

## Try it in 30 seconds

```bash
git clone https://github.com/Madhvansh/Retail-Sales-Forecasting-Engine.git
cd Retail-Sales-Forecasting-Engine
pip install -r requirements.txt && pip install -e .
python -m scripts.run_benchmark --max-series 200 --epochs 10
python -m scripts.generate_report --results results/metrics.csv
```

Runs out-of-the-box on a seeded synthetic M5-like panel — no download required.

---

<small>MIT © 2025 Madhvansh</small>
