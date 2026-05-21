#!/usr/bin/env bash
set -euo pipefail

curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/metrics | grep sentiment_api_requests_total
