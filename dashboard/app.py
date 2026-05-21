"""Streamlit analytics dashboard for the sentiment platform."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.services.model_service import get_model_service

API_URL = os.getenv("API_URL", "").rstrip("/")
REQUEST_TIMEOUT = 10

st.set_page_config(page_title="Sentiment Operations Dashboard", page_icon="chart_with_upwards_trend", layout="wide")


def api_enabled() -> bool:
    return bool(API_URL)


def api_get(path: str) -> dict[str, Any]:
    if not api_enabled():
        raise requests.RequestException("API_URL is not configured")
    response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs: Any) -> dict[str, Any]:
    if not api_enabled():
        raise requests.RequestException("API_URL is not configured")
    response = requests.post(f"{API_URL}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_resource
def local_model():
    return get_model_service()


def local_predict(review_text: str) -> dict[str, Any]:
    return local_model().predict(review_text)


def render_status() -> None:
    if api_enabled():
        try:
            health = api_get("/health")
            status = health["status"]
        except requests.RequestException:
            health = {"status": "offline", "database": "unknown", "redis": "unknown", "model_loaded": False}
            status = "offline"
    else:
        health = {"status": "local", "database": "disabled", "redis": "disabled", "model_loaded": True}
        status = "local"

    cols = st.columns(4)
    cols[0].metric("API", status.upper())
    cols[1].metric("Database", str(health["database"]).upper())
    cols[2].metric("Redis", str(health["redis"]).upper())
    cols[3].metric("Model", "LOADED" if health["model_loaded"] else "MISSING")


def render_single_prediction() -> None:
    st.subheader("Live Prediction")
    review_text = st.text_area("Review text", height=120, placeholder="Paste a customer review...")
    if st.button("Predict", type="primary", use_container_width=True) and review_text.strip():
        if api_enabled():
            try:
                result = api_post("/predict", json={"review_text": review_text})
            except requests.RequestException as exc:
                st.warning(f"API unavailable, using local Streamlit model. Details: {exc}")
                result = local_predict(review_text)
        else:
            result = local_predict(review_text)
        left, right = st.columns([1, 2])
        left.metric("Sentiment", result["sentiment"].title())
        left.metric("Confidence", f"{result['confidence']:.1%}")
        probability_df = pd.DataFrame(
            [{"sentiment": key, "probability": value} for key, value in result["probabilities"].items()]
        )
        right.plotly_chart(
            px.bar(probability_df, x="sentiment", y="probability", color="sentiment", range_y=[0, 1]),
            use_container_width=True,
        )


def render_csv_upload() -> None:
    st.subheader("CSV Upload")
    uploaded = st.file_uploader("Upload customer review CSV", type=["csv"])
    if uploaded is not None and st.button("Analyze CSV", use_container_width=True):
        if api_enabled():
            try:
                result = api_post(
                    "/upload-csv",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                )
                predictions = pd.DataFrame(result["predictions"])
            except requests.RequestException as exc:
                st.warning(f"API upload unavailable, using local Streamlit model. Details: {exc}")
                predictions = predict_uploaded_csv(uploaded)
        else:
            predictions = predict_uploaded_csv(uploaded)
        st.dataframe(predictions, use_container_width=True, hide_index=True)


def predict_uploaded_csv(uploaded: Any) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    if "review_text" not in df.columns:
        st.error("CSV must contain a review_text column.")
        return pd.DataFrame()
    rows = [local_predict(text) for text in df["review_text"].dropna().astype(str) if text.strip()]
    return pd.DataFrame(rows)


def render_analytics() -> None:
    if not api_enabled():
        st.subheader("Analytics")
        st.info(
            "Analytics history needs the FastAPI backend and PostgreSQL. "
            "Single review and CSV predictions work here."
        )
        return

    try:
        analytics = api_get("/analytics")
    except requests.RequestException as exc:
        st.warning(f"Analytics unavailable: {exc}")
        return

    st.subheader("Analytics")
    sentiment_filter = st.multiselect(
        "Filter sentiments",
        ["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )
    distribution = pd.DataFrame(
        [
            {"sentiment": sentiment, "count": count}
            for sentiment, count in analytics["sentiment_distribution"].items()
            if sentiment in sentiment_filter
        ]
    )
    trend_df = pd.DataFrame(analytics["daily_trends"])
    if not trend_df.empty:
        trend_df = trend_df[trend_df["sentiment"].isin(sentiment_filter)]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Predictions", f"{analytics['total_predictions']:,}")
    metric_cols[1].metric("Avg Confidence", f"{analytics['average_confidence']:.1%}")
    metric_cols[2].metric("Tracked Sentiments", str(len(distribution)))

    chart_cols = st.columns(2)
    if not distribution.empty:
        chart_cols[0].plotly_chart(
            px.pie(distribution, values="count", names="sentiment", hole=0.35),
            use_container_width=True,
        )
    if not trend_df.empty:
        chart_cols[1].plotly_chart(
            px.line(trend_df, x="date", y="count", color="sentiment", markers=True),
            use_container_width=True,
        )

    review_cols = st.columns(2)
    review_cols[0].caption("Top positive reviews")
    review_cols[0].dataframe(pd.DataFrame(analytics["top_positive_reviews"]), use_container_width=True)
    review_cols[1].caption("Top negative reviews")
    review_cols[1].dataframe(pd.DataFrame(analytics["top_negative_reviews"]), use_container_width=True)


def main() -> None:
    st.title("Sentiment Operations Dashboard")
    render_status()
    tab_live, tab_upload, tab_analytics = st.tabs(["Predict", "Upload CSV", "Analytics"])
    with tab_live:
        render_single_prediction()
    with tab_upload:
        render_csv_upload()
    with tab_analytics:
        render_analytics()


if __name__ == "__main__":
    main()
