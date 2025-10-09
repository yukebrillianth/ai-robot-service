FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# -------- Environment Setup --------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA="1"
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV MEDIAPIPE_MODEL_PATH=/app/.cache/mediapipe
ENV ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json
ENV TMPDIR=/app/.cache
RUN mkdir -p /app/.cache/mediapipe /app/.cache/ultralytics && chown -R appuser:appuser /app/.cache

# -------- System Dependencies --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    libhdf5-dev libqt5core5a libqt5gui5 libqt5widgets5 libgstreamer-plugins-base1.0-dev \
    python3-dev python3-pip python3-setuptools python3-wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Python Packages --------
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    opencv-python-headless \
    onnxruntime-gpu \
    ultralytics \
    websockets \
    fastapi \
    uvicorn[standard]


# -------- Copy Project --------
COPY . .

# -------- Download Model Sekali (opsional) --------
RUN python3 - <<'PY'
try:
    from ultralytics import YOLO
    for m in ['yolo11m.pt', 'yolov8n.pt', 'yolov8m.pt']:
        YOLO(m)
except Exception as e:
    print(f"⚠️  Model download skipped: {e}")
PY

# -------- Non-root User --------
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# -------- Port & Health Check --------
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# -------- Default Command --------
CMD ["python", "server/production_server.py"]
