# Contributing

Thanks for your interest in improving the Retail Forecasting Engine!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
```

## Before you open a PR

```bash
ruff check src scripts app tests      # lint
pytest -q                             # tests
```

- Keep functions small and typed; match the surrounding style.
- New models go through `Forecaster` + the registry, with a unit test.
- New metrics belong in `src/evaluation/metrics.py` with a test that pins a
  known value.
- Don't commit data (`data/`) or generated artifacts (`results/`).

## Commit messages

Use imperative, descriptive subjects (e.g. "Add TSB quantile calibration").
