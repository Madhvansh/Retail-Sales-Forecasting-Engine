.PHONY: install install-dev data benchmark report dashboard api test lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,serve,foundation]"

data:
	python -m scripts.download_data

benchmark:
	python -m scripts.run_benchmark --config config/config.yaml

report:
	python -m scripts.generate_report --results results/metrics.csv

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src scripts app tests

clean:
	rm -rf results/figures/*.png results/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
