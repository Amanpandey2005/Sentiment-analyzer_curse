# Docker Notes

The root `Dockerfile` is a multi-stage production image used for both the FastAPI API and Streamlit dashboard. Runtime behavior is selected by the Compose command.

- `docker-compose.yml` runs the local full stack.
- `docker-compose.prod.yml` pulls the API/dashboard image from ECR for EC2 deployments.
- Health checks are defined in both Dockerfile and Compose for API readiness.
