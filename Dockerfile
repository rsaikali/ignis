# syntax=docker/dockerfile:1
# ha_ingest image: lightweight, no TensorFlow.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- deps layer: cached unless pyproject changes ----------------------------
# Only pyproject is a build input here (a stub README + stub package satisfy
# hatchling). README/src changes do NOT bust this heavy layer.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/ignis && touch src/ignis/__init__.py && echo "stub" > README.md \
    && pip install --upgrade pip && pip install . \
    && rm -rf src README.md

# --- code layer: fast (deps already installed) ------------------------------
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .

CMD ["python", "-m", "ignis.ha_ingest"]
