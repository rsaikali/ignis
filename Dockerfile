# syntax=docker/dockerfile:1
# ha_ingest image: lightweight, no TensorFlow.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- deps layer: cached unless pyproject changes ----------------------------
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/ignis && touch src/ignis/__init__.py \
    && pip install --upgrade pip && pip install . \
    && rm -rf src

# --- code layer: fast -------------------------------------------------------
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .

CMD ["python", "-m", "ignis.ha_ingest"]
