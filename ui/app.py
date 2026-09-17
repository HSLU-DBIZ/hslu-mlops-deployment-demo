"""Streamlit UI: streamlit run ui/app.py

Default view: the batch job's predictions file. On demand: POST /predict on the API.
Both show the model version and the feature time next to the numbers.
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

PREDICTIONS_URI = os.getenv("PREDICTIONS_URI", "predictions/latest.csv")
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.title("Lucerne temperature forecast")


@st.cache_data(ttl=600)  # re-read at most every 10 min
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_URI, parse_dates=["valid_time", "feature_time", "created_at"])


if "://" in PREDICTIONS_URI or Path(PREDICTIONS_URI).exists():  # batch: read the table
    df = load_predictions()
    st.line_chart(df, x="valid_time", y="temperature_c", y_label="°C")
    st.caption(
        f"model v{df.model_version.iloc[-1]} · features as of {df.feature_time.max():%d.%m. %H:%M} "
        f"UTC ({df.feature_source.iloc[-1]}) · created {df.created_at.max():%d.%m. %H:%M} UTC"
    )
else:
    st.info("No batch predictions yet. Run: python -m lucerne_temp.inference.batch")

hours = st.slider("Hours ahead", 1, 24, 3)
if st.button("Forecast"):  # on-demand: call the API
    try:
        r = httpx.post(f"{API_URL}/predict", json={"hours_ahead": hours}, timeout=30)
        r.raise_for_status()
    except httpx.HTTPError as exc:  # timeout 30 s: the first call after idle is a cold start
        st.error(f"API not reachable at {API_URL}: {exc}")
    else:
        f = r.json()
        st.metric(f"Temperature at {f['valid_time']}", f"{f['temperature_c']:.1f} °C")
        st.caption(
            f"model v{f['model_version']} · features as of {f['feature_time']} "
            f"({f['feature_source']})"
        )

st.caption("Weather data by Open-Meteo.com (CC BY 4.0)")
