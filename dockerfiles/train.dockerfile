FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2. Install Python, Git, and generic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Install 'uv' manually
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4. Copy configuration files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /

# 5. Tell uv to use Python 3.11 (since Ubuntu 22.04 defaults to 3.10)
RUN uv python install 3.11

# 6. Install dependencies
RUN uv sync --frozen --no-cache --no-install-project

# 7. Copy your source code and configurations
COPY src ./src
COPY configs ./configs
COPY README.md ./README.md
COPY LICENSE ./LICENSE

# 8. Copy DVC files (Critical for data loading!)
COPY .dvc ./.dvc
COPY *.dvc ./

# 9. Set the entrypoint
ENTRYPOINT ["uv", "run", "src/ml_ops_project/train.py"]
