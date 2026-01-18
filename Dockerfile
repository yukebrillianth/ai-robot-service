# =============================================================================
# Dockerfile untuk NVIDIA GH200 (ARM64/aarch64) + OpenShift
# =============================================================================

# Base image ARM64 untuk GH200 Grace Hopper dengan support sm_90 (Hopper)
# PyTorch 2.6 + CUDA 12.6.3 + cuDNN 9.6 + TensorRT 10.7
# Lihat: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-12.html
FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json \
    YOLO_OFFLINE=True \
    TMPDIR=/app/.cache

# Install system dependencies yang diperlukan OpenCV & Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Buat folder cache di awal agar permission benar
RUN mkdir -p /app/.cache/matplotlib /app/.cache/ultralytics /app/.cache/mediapipe /app/models

COPY requirements.txt .

# Instalasi Python packages
# PENTING: JANGAN uninstall opencv dari base image NVIDIA
# Base image sudah punya opencv yang compatible dengan CUDA + arsitektur ARM64
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir --no-deps ultralytics && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY . .

# Fix permission untuk OpenShift
RUN chgrp -R 0 /app && chmod -R g+rwX /app

# Download model saat build time (offline mode sudah aktif via ENV)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" && \
    mv yolo11m.pt /app/models/ 2>/dev/null || true

# USER 1001 sesuai standar keamanan OpenShift
USER 1001

EXPOSE 8000

# Healthcheck untuk readiness/liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server/server2.py"]