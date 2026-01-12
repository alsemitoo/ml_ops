FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

# Default entrypoint for data operations
ENTRYPOINT ["uv", "run", "python", "src/ml_ops_project/data.py"]