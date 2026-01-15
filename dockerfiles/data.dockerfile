FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /

RUN uv sync --locked --no-cache --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

ENTRYPOINT ["uv", "run", "python", "src/ml_ops_project/data.py"]