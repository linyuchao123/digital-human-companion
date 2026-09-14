FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY apps ./apps
COPY packages ./packages
COPY services ./services
ARG APP_EXTRAS=cloud,ml,speech,rag,vision
RUN pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[${APP_EXTRAS}]"
COPY . .
RUN mkdir -p data/kb data/processed models/asr cache \
    && useradd --uid 10001 --create-home appuser \
    && chown -R appuser:appuser data models/asr cache
ENV HF_HOME=/app/cache/huggingface MODELSCOPE_CACHE=/app/cache/modelscope
USER appuser
EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8800/api/health/ready || exit 1
CMD ["python", "-m", "uvicorn", "apps.api.integrated_server:app", "--host", "0.0.0.0", "--port", "8800", "--workers", "1", "--ws-max-size", "3145728"]
