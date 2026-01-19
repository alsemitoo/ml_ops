FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --frozen --no-install-project --no-dev

COPY src /app/src
COPY configs /app/configs/
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY frontend.py /app/frontend.py

EXPOSE 8501

ENTRYPOINT ["uv", "run","streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.serverAddress=localhost"]