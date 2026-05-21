from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_analyzer.dashboard_backend import (
    DEFAULT_MODEL_PATH,
    analyze_dataframe,
    ensure_model,
    label_distribution,
    load_input_data,
    summarize_predictions,
)


ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "sample_reviews.csv"


st.set_page_config(
    page_title="E-Commerce Sentiment Analyzer",
    page_icon=":material/shopping_cart:",
    layout="wide",
)


@st.cache_resource
def get_model():
    return ensure_model(SAMPLE_DATA, DEFAULT_MODEL_PATH)


def render_metric_cards(summary: dict[str, object]) -> None:
    total, positive, neutral, negative, avg_confidence = st.columns(5)
    total.metric("Reviews", f"{summary['total_reviews']:,}")
    positive.metric("Positive", f"{summary['positive_count']:,}")
    neutral.metric("Neutral", f"{summary['neutral_count']:,}")
    negative.metric("Negative", f"{summary['negative_count']:,}")
    avg_confidence.metric("Avg confidence", f"{summary['average_confidence']:.1%}")


def render_single_review(model) -> None:
    st.subheader("Analyze one review")
    review_text = st.text_area(
        "Review text",
        placeholder="Paste a customer review here...",
        height=130,
        label_visibility="collapsed",
    )

    if st.button("Analyze review", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Enter a review before running sentiment analysis.")
            return

        probabilities = model.predict_proba(review_text)
        sentiment = max(probabilities, key=probabilities.get)
        confidence = probabilities[sentiment]

        result, chart = st.columns([1, 2])
        result.metric("Predicted sentiment", sentiment.title())
        result.metric("Confidence", f"{confidence:.1%}")
        chart.bar_chart(pd.DataFrame([probabilities]).T.rename(columns={0: "probability"}))


def render_batch_analysis(model) -> None:
    st.subheader("Analyze review dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    data_source = uploaded_file if uploaded_file is not None else SAMPLE_DATA

    try:
        df = load_input_data(data_source)
    except ValueError as exc:
        st.error(str(exc))
        return

    analyzed = analyze_dataframe(df, model)
    summary = summarize_predictions(analyzed)
    render_metric_cards(summary)

    chart_data = label_distribution(analyzed)
    st.bar_chart(chart_data.set_index("sentiment"))

    st.dataframe(
        analyzed,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        },
    )

    csv_bytes = analyzed.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions",
        data=csv_bytes,
        file_name="sentiment_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.title("E-Commerce Sentiment Analyzer")
    st.caption("Classify customer reviews, inspect confidence, and export predictions.")

    model = get_model()
    tab_single, tab_batch = st.tabs(["Single Review", "Batch Dataset"])

    with tab_single:
        render_single_review(model)

    with tab_batch:
        render_batch_analysis(model)


if __name__ == "__main__":
    main()
