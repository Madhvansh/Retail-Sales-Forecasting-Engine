# Deployment Guide

This project ships three runnable surfaces: a **batch benchmark**, a
**REST API** (FastAPI) and an **interactive dashboard** (Streamlit). Below are
the supported deployment paths.

---

## 1. Local (Python virtualenv)

```bash
git clone https://github.com/Madhvansh/retail-sales-forecasting-engine.git
cd retail-sales-forecasting-engine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# (optional) real M5 data — otherwise synthetic data is generated automatically
python -m scripts.download_data        # needs ~/.kaggle/kaggle.json

# run the benchmark + build the report
python -m scripts.run_benchmark --config config/config.yaml
python -m scripts.generate_report --results results/metrics.csv

# serve
uvicorn app.api:app --host 0.0.0.0 --port 8000      # API
streamlit run app/dashboard.py                       # dashboard
```

> **GPU**: PatchTST and Chronos-Bolt auto-detect CUDA (`device: auto`). On CPU
> they still run; reduce `data.max_series` for a quick pass.

---

## 2. Docker (single image, two services)

```bash
docker compose build
docker compose up            # API on :8000, dashboard on :8501

# one-off benchmark inside the container
docker compose run --rm api python -m scripts.run_benchmark --max-series 200
```

The `results/` and `data/` directories are bind-mounted so artifacts persist
on the host.

---

## 3. Cloud container platforms

The image is a standard Python service and runs anywhere that accepts a
container:

| Platform            | Notes                                                        |
|---------------------|-------------------------------------------------------------|
| AWS ECS / Fargate   | Push the image to ECR; expose port 8000 behind an ALB.      |
| Google Cloud Run    | `gcloud run deploy --image ... --port 8000`. Stateless API. |
| Azure Container Apps| Set `targetPort: 8000`; scale-to-zero friendly.             |
| Fly.io / Render     | Point at the Dockerfile; set the start command per service. |

### Environment variables

| Variable            | Purpose                                            |
|---------------------|----------------------------------------------------|
| `HF_HOME`           | Hugging Face cache dir for Chronos-Bolt weights.   |
| `CHRONOS_MODEL`     | Override the default `amazon/chronos-bolt-small`.   |
| `PYTHONUNBUFFERED`  | Stream logs (set to `1`).                           |

> The first Chronos-Bolt request downloads weights from the Hugging Face Hub,
> so the API needs outbound network access (or a pre-warmed `HF_HOME` volume).

---

## 4. Health checks & smoke test

```bash
curl -s localhost:8000/health
curl -s localhost:8000/models
curl -s localhost:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"history":[3,0,0,5,0,2,0,4,1,0,0,3], "horizon":7, "model":"croston_sba"}'
```

---

## 5. Scheduled re-benchmarking

Run the benchmark on a schedule (cron / GitHub Actions / Cloud Scheduler) to
refresh `results/metrics.csv`; the dashboard reads it live on reload.

```cron
0 3 * * 1  cd /opt/rfe && python -m scripts.run_benchmark && python -m scripts.generate_report
```
