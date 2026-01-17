FROM nvcr.io/nvidia/pytorch:24.02-py3

# -------- Environment Setup --------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV MEDIAPIPE_MODEL_PATH=/app/.cache/mediapipe
ENV ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json
ENV TMPDIR=/app/.cache

# -------- System Dependencies --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    libhdf5-dev libqt5core5a libqt5gui5 libqt5widgets5 libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Prepare Cache Directories --------
RUN mkdir -p /app/.cache/mediapipe /app/.cache/ultralytics

# -------- Python Packages --------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    opencv-python-headless \
    ultralytics \
    websockets \
    fastapi \
    uvicorn[standard] && \
    pip install --no-cache-dir "numpy<2.0" --force-reinstall

# -------- Copy Project --------
COPY . .

# -------- OpenShift Permission Fix --------
RUN chgrp -R 0 /app && \
    chmod -R g+rwX /app

# -------- Pre-download Models (Opsional) --------
RUN python3 - <<'PY'
try:
    from ultralytics import YOLO
    # Download model ke folder cache
    for m in ['yolo11m.pt', 'yolov8n.pt', 'yolov8m.pt']:
        YOLO(m)
except Exception as e:
    print(f"⚠️  Model download skipped: {e}")
PY

# -------- Switch User --------
USER 1001

EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# -------- Default Command --------
CMD ["python", "server/server2.py"]
