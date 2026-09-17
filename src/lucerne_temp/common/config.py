"""Settings without secrets. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(os.getenv("LUCERNE_TEMP_ROOT", Path(__file__).resolve().parents[3]))

SNAPSHOT_PATH = ROOT / "data" / "temperature_snapshot.csv"
REGISTRY_PATH = ROOT / "registry" / "registry.json"
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
PREDICTIONS_URI = os.getenv("PREDICTIONS_URI", str(ROOT / "predictions" / "latest.csv"))

MODEL_NAME = "temp-forecaster"
ALIAS = "champion"

# Lucerne
LATITUDE = 47.05
LONGITUDE = 8.31

OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LIVE_TIMEOUT_S = float(os.getenv("LIVE_TIMEOUT_S", "3"))
LIVE_CACHE_S = float(os.getenv("LIVE_CACHE_S", "300"))
