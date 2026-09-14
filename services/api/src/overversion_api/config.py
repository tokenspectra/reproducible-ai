"""Settings, read from the environment (Railway variables in production)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Inference box
    infer_base_url: str = Field(default="https://infer.overversion.com")
    infer_api_key: str = Field(default="")
    infer_baseline_path: str = Field(default="/baseline")
    infer_model: str = Field(default="qwen3-8b")
    infer_timeout_s: float = Field(default=300.0)

    # OpenRouter
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="qwen/qwen3-8b")

    # Status poller
    status_interval_s: float = Field(default=10.0)
    status_timeout_s: float = Field(default=5.0)

    # Storage and identity
    database_url: str = Field(default="sqlite+aiosqlite:///./overversion.db")
    token_signing_seed_hex: str = Field(default="")
    app_origin: str = Field(default="https://app.overversion.com")
    frontend_shared_secret: str = Field(default="")

    # Evidence
    top_logprobs_k: int = Field(default=32)

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        """Railway hands out `postgresql://`; SQLAlchemy needs an async driver spelled out.

        Normalising here means the deployment can use Railway's reference
        variable verbatim and nobody has to remember to rewrite the scheme.
        """
        for prefix in ("postgresql+asyncpg://", "sqlite+aiosqlite://"):
            if v.startswith(prefix):
                return v
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://") :]
        if v.startswith("sqlite://"):
            return "sqlite+aiosqlite://" + v[len("sqlite://") :]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
