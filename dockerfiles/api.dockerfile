FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /

RUN uv sync --frozen --no-install-project

COPY src ./src
COPY models ./models
COPY configs ./configs
COPY README.md ./README.md
COPY LICENSE ./LICENSE

RUN uv sync --frozen

# Cloud Run automatically sets PORT environment variable at runtime
EXPOSE 8080

ENTRYPOINT ["uv", "run", "uvicorn", "src.ml_ops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
