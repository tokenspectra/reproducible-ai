# Build context is the repo root: the API depends on packages/core by path.
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
RUN uv venv /opt/venv

# Dependency layer: only the manifests, so edits to source do not reinstall the world.
COPY packages/core/pyproject.toml packages/core/README.md ./packages/core/
COPY services/api/pyproject.toml ./services/api/
RUN mkdir -p packages/core/src/reproducible_ai services/api/src/overversion_api \
 && touch packages/core/src/reproducible_ai/__init__.py services/api/src/overversion_api/__init__.py \
 && uv pip install --no-cache -e ./packages/core -e ./services/api

# Source layer.
COPY packages/core ./packages/core
COPY services/api ./services/api

EXPOSE 8000
# Railway sets $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "exec uvicorn overversion_api.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
