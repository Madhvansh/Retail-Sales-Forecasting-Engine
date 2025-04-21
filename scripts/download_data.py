"""Fetch / verify the M5 (Walmart) competition dataset.

The M5 data lives on Kaggle ("M5 Forecasting - Accuracy"). Kaggle requires
authentication, so this script supports two paths:

1. **Kaggle API** (preferred): set up `~/.kaggle/kaggle.json` and run
   ``python -m scripts.download_data``. The competition files are downloaded
   and unzipped into ``data/raw``.

2. **Manual**: download the zip from Kaggle, drop the CSVs into ``data/raw``
   and this script will simply verify they are present.

The three files the pipeline expects are:
    - ``sales_train_validation.csv``
    - ``calendar.csv``
    - ``sell_prices.csv``
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from src.utils import get_logger

log = get_logger("download")

COMPETITION = "m5-forecasting-accuracy"
REQUIRED_FILES = [
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
]


def verify(raw_dir: Path) -> bool:
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        log.warning("Missing M5 files in %s: %s", raw_dir, ", ".join(missing))
        return False
    log.info("All required M5 files are present in %s", raw_dir)
    return True


def download_via_kaggle(raw_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        log.error(
            "kaggle package not installed. Install it (`pip install kaggle`) and "
            "configure ~/.kaggle/kaggle.json, or place the CSVs in %s manually.",
            raw_dir,
        )
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    log.info("Downloading competition '%s' into %s ...", COMPETITION, raw_dir)
    api.competition_download_files(COMPETITION, path=str(raw_dir), quiet=False)

    for zip_path in raw_dir.glob("*.zip"):
        log.info("Extracting %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/verify the M5 dataset.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check that the CSVs exist; do not attempt a download.",
    )
    args = parser.parse_args(argv)

    args.raw_dir.mkdir(parents=True, exist_ok=True)

    if verify(args.raw_dir):
        return 0
    if args.verify_only:
        return 1

    download_via_kaggle(args.raw_dir)
    return 0 if verify(args.raw_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
