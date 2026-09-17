"""The serving contract as pydantic v2 models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FeatureSource = Literal["live", "snapshot"]


class ForecastRequest(BaseModel):
    hours_ahead: int = Field(default=1, ge=1, le=24)


class Forecast(BaseModel):
    valid_time: datetime
    temperature_c: float
    model_version: str
    feature_time: datetime
    feature_source: FeatureSource


class Health(BaseModel):
    status: str
    model_version: str
    feature_time: datetime
    feature_source: FeatureSource
