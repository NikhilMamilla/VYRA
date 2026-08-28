# Built from the repository root so the image can include the `vyra_ml`
# inference package and the model bundle:
#   docker build -f backend/Dockerfile -t vyra-backend .
# (docker-compose.yml sets context: . for this service.)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Python deps (cached until requirements change). opencv-headless needs a few
# shared libs even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. The inference package, without its (heavy, dev-only) extra deps -- the
# runtime ones are already in requirements.txt.
COPY ml/pyproject.toml ml/README.md /tmp/ml/
COPY ml/vyra_ml /tmp/ml/vyra_ml
RUN pip install --no-cache-dir --no-deps /tmp/ml && rm -rf /tmp/ml

# 3. Application code and the model bundle.
COPY backend/app ./app
COPY ml/artifacts/vyra-quality-model-v1 ./model

ENV MODEL_PATH=/app/model \
    REQUIRE_ANALYZER=true

RUN useradd --create-home --uid 1000 vyra \
    && mkdir -p /app/data/uploads \
    && chown -R vyra:vyra /app
USER vyra

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
