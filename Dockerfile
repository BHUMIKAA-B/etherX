# syntax=docker/dockerfile:1

# ── Stage 1: dependency + model cache ─────────────────────────────────────────
FROM python:3.10-slim AS base

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 libgomp1\
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-prod.txt* ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn
    
ENV OMP_NUM_THREADS=1
ENV KMP_INIT_AT_FORK=FALSE
ENV FLAGS_use_mkldnn=false
# Pre-download PaddleOCR models into the image layer.
# This ensures container startup is fast (no network needed at runtime).
#RUN python -c "\
#from paddleocr import PaddleOCR; \
#print('Pre-loading PaddleOCR models...'); \
#PaddleOCR(use_textline_orientation=True, lang='en', device='cpu', enable_mkldnn=False); \
#print('Done.')"

# ── Stage 2: application ───────────────────────────────────────────────────────
FROM base

WORKDIR /app

COPY . .

# Create runtime dirs
RUN mkdir -p uploads outputs

EXPOSE 8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
