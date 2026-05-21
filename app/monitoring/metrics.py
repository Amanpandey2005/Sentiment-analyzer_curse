"""Prometheus metrics setup."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "sentiment_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "sentiment_api_request_duration_seconds",
    "API request latency",
    ["method", "endpoint"],
)
PREDICTION_COUNT = Counter(
    "sentiment_predictions_total",
    "Total sentiment predictions",
    ["sentiment"],
)


async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Collect latency and request count metrics."""

    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(request.method, endpoint).observe(latency)
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    return response


def metrics_response() -> Response:
    """Return Prometheus exposition text."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
