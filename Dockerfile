# ha_ingest image: lightweight, no TensorFlow.
# The inference worker (engine + TF CPU) gets its own image once the engine is
# wired to the ha_samples schema -- see docker-compose.yml "worker" profile.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching. ha_ingest needs only the base deps.
COPY pyproject.toml README.md ./
COPY nilm/ ./nilm/
COPY ha_ingest/ ./ha_ingest/
RUN pip install --upgrade pip && pip install .

CMD ["python", "-m", "ha_ingest"]
