# Dockerized ML Sentiment Analysis Platform

A production-style MLOps and DevOps portfolio project for customer review sentiment analysis. It includes a FastAPI inference service, scikit-learn training pipeline, Streamlit analytics dashboard, PostgreSQL persistence, Redis rate limiting, Prometheus/Grafana monitoring, Docker Compose, NGINX, GitHub Actions CI/CD, and Terraform AWS infrastructure.

## Architecture

```text
Client / Browser
  -> NGINX reverse proxy
    -> Streamlit dashboard
    -> FastAPI REST API
      -> TF-IDF + Logistic Regression model
      -> PostgreSQL predictions table
      -> Redis rate limiter
      -> Prometheus /metrics
Prometheus -> Grafana dashboards
GitHub Actions -> AWS ECR -> EC2 Docker Compose deployment
```

## Project Structure

```text
app/                    FastAPI app, auth, database, middleware, services, metrics
ml/                     Training, preprocessing, evaluation, saved model artifacts
dashboard/              Streamlit analytics dashboard
tests/                  Unit, API, integration, and Docker config tests
nginx/                  Reverse proxy and HTTPS-ready configuration
monitoring/             Prometheus and Grafana provisioning
scripts/                EC2 bootstrap, deploy, and health-check scripts
terraform/              AWS EC2, ECR, S3, IAM, CloudWatch, security groups
.github/workflows/      CI/CD pipeline
docker-compose.yml      Local full-stack deployment
docker-compose.prod.yml EC2 production deployment
Dockerfile              Multi-stage production image
```

## Features

- Single and batch sentiment prediction: positive, negative, neutral
- CSV upload endpoint and dashboard workflow
- Review summaries generated during inference
- PostgreSQL storage for every prediction
- Analytics API and Streamlit dashboard with distribution, confidence, trends, filters, and top reviews
- JWT token utility, Redis-backed rate limiting, structured JSON logging
- Prometheus metrics for request count, latency, errors, and prediction counts
- Grafana dashboard provisioning with application, CPU, and memory panels
- Multi-stage Docker builds with non-root runtime user and health checks
- GitHub Actions pipeline for tests, linting, ECR image push, and EC2 deployment
- Terraform templates for AWS infrastructure

## Run Locally

```bash
docker compose up --build
```

Services:

- Dashboard: `http://localhost`
- API: `http://localhost/api`
- Direct API docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` with `admin/admin`
- cAdvisor: `http://localhost:8080`

The API trains `ml/saved_models/sentiment_model.joblib` from `data/sample_reviews.csv` automatically if the artifact is missing.

## API Examples

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text":"Excellent quality and fast delivery."}'
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/analytics
curl http://localhost:8000/metrics
```

More endpoint details are in [docs/API.md](docs/API.md).

## Train and Evaluate Locally

```bash
python -m ml.training.train --data data/sample_reviews.csv --model-out ml/saved_models/sentiment_model.joblib
python -m ml.evaluation.evaluate --data data/sample_reviews.csv --model ml/saved_models/sentiment_model.joblib
```

CSV files should include `review_text` and optionally `sentiment` for training.

## Test

```bash
pip install -r requirements.txt
pytest -q
ruff check app ml dashboard tests
```

## CI/CD Flow

```text
Developer pushes to main
  -> GitHub Actions installs dependencies
  -> Runs Ruff and Pytest
  -> Builds Docker image
  -> Pushes latest and SHA tags to AWS ECR
  -> SSHes into EC2
  -> Pulls latest image
  -> Restarts Docker Compose services
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for AWS setup, IAM, EC2, ECR, CloudWatch, HTTPS, and GitHub secrets.
