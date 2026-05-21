# API Documentation

Base URL:

- Local direct API: `http://localhost:8000`
- NGINX proxy: `http://localhost/api`

Interactive OpenAPI documentation is available at `/docs` and ReDoc at `/redoc`.

## Endpoints

### `GET /health`

Returns dependency status for the API, database, Redis, and model artifact.

### `GET /metrics`

Returns Prometheus metrics including request counts, latency histograms, and prediction totals.

### `POST /predict`

Request:

```json
{
  "review_text": "Excellent product and fast delivery."
}
```

Response:

```json
{
  "review_text": "Excellent product and fast delivery.",
  "sentiment": "positive",
  "confidence": 0.82,
  "probabilities": {
    "negative": 0.08,
    "neutral": 0.10,
    "positive": 0.82
  },
  "summary": "Positive review summary: Excellent product and fast delivery."
}
```

### `POST /batch-predict`

Request:

```json
{
  "reviews": ["Great build quality", "The product broke immediately"]
}
```

### `POST /upload-csv`

Multipart upload with a CSV file containing a `review_text` column. Predictions are persisted to PostgreSQL.

### `GET /analytics`

Returns total predictions, sentiment distribution, average confidence, daily trends, and top positive/negative reviews.

### `POST /auth/token`

Issues a demo JWT for environments where you want Bearer-token calls. The current portfolio configuration allows anonymous inference but validates tokens when supplied.
