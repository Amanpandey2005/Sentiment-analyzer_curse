from __future__ import annotations

from pathlib import Path


def test_dockerfile_contains_production_controls() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim AS builder" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_compose_defines_required_services() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for service in ["api:", "dashboard:", "postgres:", "redis:", "nginx:", "prometheus:", "grafana:"]:
        assert service in compose
