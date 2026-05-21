"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core_config import get_settings
from app.database.session import init_db
from app.monitoring.metrics import metrics_middleware
from app.services.model_service import get_model_service
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize database and warm the model."""

    init_db()
    get_model_service()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Dockerized ML sentiment analysis platform with MLOps and DevOps workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(metrics_middleware)
app.include_router(router, prefix=settings.api_prefix)
