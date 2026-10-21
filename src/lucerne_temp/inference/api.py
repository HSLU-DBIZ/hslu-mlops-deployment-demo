"""On-demand inference: uvicorn lucerne_temp.inference.api:app --port 8080"""

import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from lucerne_temp.features.build import serving_features
from lucerne_temp.features.ingest import latest_temps
from lucerne_temp.inference.model import load_champion
from lucerne_temp.inference.schemas import (
    Forecast,
    ForecastRequest,
    Health,
    PredictInstances,
    Predictions,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["version"], state["model"] = load_champion()  # once, at startup
    yield
    state.clear()


app = FastAPI(title="lucerne-temp", lifespan=lifespan)


@app.get("/health")
def health() -> Health:
    temps, source = latest_temps()
    return Health(status="ok", model_version=state["version"],
                  feature_time=temps.index[-1], feature_source=source)  # fmt: skip


@app.post("/predict")
def predict(req: ForecastRequest) -> Forecast:
    start = time.perf_counter()
    temps, source = latest_temps()
    X, valid, ft = serving_features(temps, req.hours_ahead)
    y = round(float(state["model"].predict(X)[0]), 2)

    # Cloud Run: one JSON line on stdout = one structured log entry
    print(json.dumps({
        "event": "prediction",
        "ts": datetime.now(UTC).isoformat(),
        "model_version": state["version"],
        "feature_time": ft.isoformat(),
        "feature_source": source,
        "features": X[state["model"].features].iloc[0].round(3).to_dict(),  # what the model saw
        "hours_ahead": req.hours_ahead,
        "temperature_c": y,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }), flush=True)  # fmt: skip
    return Forecast(valid_time=valid, temperature_c=y, model_version=state["version"],
                    feature_time=ft, feature_source=source)  # fmt: skip


# Agent Platform (Vertex AI) contract: {"instances": [...]} in, {"predictions": [...]} out
@app.post("/v1/predict")
def predict_instances(req: PredictInstances) -> Predictions:
    return Predictions(predictions=[predict(r) for r in req.instances])  # one log line each
