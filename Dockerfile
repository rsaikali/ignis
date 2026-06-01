# ha_ingest image: lightweight, no TensorFlow.
# The inference worker (engine + TF CPU) gets its own image once the engine is
# wired to the ha_samples schema -- see docker-compose.yml "worker" profile.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ha_ingest needs only the base deps. The package lives under src/ignis.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

CMD ["python", "-m", "ignis.ha_ingest"]
