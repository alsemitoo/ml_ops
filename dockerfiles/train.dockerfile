FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /

RUN uv sync --locked --no-cache --no-install-project

# Now copy your source
COPY src ./src
COPY configs ./configs
COPY README.md ./README.md
COPY LICENSE ./LICENSE

# Install your project into the env (and verify lock)
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/ml_ops_project/train.py"]
