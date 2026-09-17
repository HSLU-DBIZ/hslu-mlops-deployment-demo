"""The model object that gets saved, loaded and served."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Forecaster:
    """Predicts the temperature change until the valid time and adds it to temp_now.

    Trees cannot extrapolate. The change until the valid time stays in a familiar range
    when the season moves on; the temperature itself does not.
    """

    estimator: Any
    features: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        self.estimator.fit(X[self.features], y - X["temp_now"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["temp_now"].to_numpy() + self.estimator.predict(X[self.features])
