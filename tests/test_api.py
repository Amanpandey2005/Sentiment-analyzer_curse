from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_status() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_predict_endpoint_returns_sentiment() -> None:
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"review_text": "Excellent quality and fast delivery"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment"] in {"positive", "neutral", "negative"}
    assert 0 <= payload["confidence"] <= 1
    assert "summary" in payload


def test_batch_predict_endpoint_returns_predictions() -> None:
    with TestClient(app) as test_client:
        response = test_client.post(
            "/batch-predict",
            json={"reviews": ["Great product", "Poor quality"]},
        )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentiment_api_requests_total" in response.text
