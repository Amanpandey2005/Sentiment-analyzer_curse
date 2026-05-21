"""Centralized application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    app_name: str = "ML Sentiment Analysis Platform"
    environment: str = "local"
    log_level: str = "INFO"
    api_prefix: str = ""

    database_url: str = Field(
        "postgresql+psycopg://sentiment:sentiment@postgres:5432/sentiment",
        validation_alias="DATABASE_URL",
    )
    redis_url: str = Field("redis://redis:6379/0", validation_alias="REDIS_URL")

    jwt_secret_key: str = Field("change-me-in-production", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    api_rate_limit_per_minute: int = 120

    model_path: Path = Path("ml/saved_models/sentiment_model.joblib")
    training_data_path: Path = Path("data/sample_reviews.csv")
    upload_dir: Path = Path("data/uploads")
    cors_origins: list[str] = ["*"]

    aws_region: str = "us-east-1"
    s3_model_bucket: str | None = None
    cloudwatch_log_group: str = "/ml-sentiment-platform/api"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for dependency injection."""

    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
