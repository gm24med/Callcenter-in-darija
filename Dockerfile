# ---------- Base image ----------
FROM python:3.9-slim

# ---------- Env vars ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TRANSFORMERS_CACHE=/tmp/app_cache/transformers_cache \
    HF_HOME=/tmp/app_cache/hf_home \
    XDG_CACHE_HOME=/tmp/app_cache/xdg_cache \
    HUGGINGFACE_HUB_CACHE=/tmp/app_cache/hf_hub_cache \
    TORCH_HOME=/tmp/app_cache/torch_home \
    TOKENIZERS_PARALLELISM=false

# ---------- Workdir ----------
WORKDIR /app

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
        libblas-dev liblapack-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Folders ----------
RUN mkdir -p /tmp/app_cache/transformers_cache \
             /tmp/app_cache/hf_home \
             /tmp/app_cache/xdg_cache \
             /tmp/app_cache/hf_hub_cache \
             /tmp/app_cache/torch_home \
             /tmp/app_data/conversations \
             /tmp/templates /tmp/static \
    && chmod -R 777 /tmp/app_cache /tmp/app_data /tmp/templates /tmp/static

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# ---------- App code ----------
COPY app.py .

# ---------- Health‑check ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# ---------- Port (informational) ----------
EXPOSE 7860

# ---------- Start command ----------
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --log-level info --timeout-keep-alive 65"]
