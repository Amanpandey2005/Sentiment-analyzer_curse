"""Streamlit analytics dashboard for the sentiment API."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
REQUEST_TIMEOUT = 10

st.set_page_config(page_title="Sentiment Operations Dashboard", page_icon="chart_with_upwards_trend", layout="wide")


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.post(f"{API_URL}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def render_status() -> None:
    try:
        health = api_get("/health")
        status = health["status"]
    except requests.RequestException:
        health = {"status": "offline", "database": "unknown", "redis": "unknown", "model_loaded": False}
        status = "offline"

    cols = st.columns(4)
    cols[0].metric("API", status.upper())
    cols[1].metric("Database", str(health["database"]).upper())
    cols[2].metric("Redis", str(health["redis"]).upper())
    cols[3].metric("Model", "LOADED" if health["model_loaded"] else "MISSING")


def render_single_prediction() -> None:
    st.subheader("Live Prediction")
    review_text = st.text_area("Review text", height=120, placeholder="Paste a customer review...")
    if st.button("Predict", type="primary", use_container_width=True) and review_text.strip():
        try:
            result = api_post("/predict", json={"review_text": review_text})
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
            return
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
        try:
            result = api_post(
                "/upload-csv",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            )
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
            return
        predictions = pd.DataFrame(result["predictions"])
        st.dataframe(predictions, use_container_width=True, hide_index=True)


def render_analytics() -> None:
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
