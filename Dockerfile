FROM nvcr.io/nvidia/pytorch:25.12-py3

# -------- Environment Setup --------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# NGC Image sudah punya CUDA environment, tapi ini harmless
ENV FORCE_CUDA="1" 
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV MEDIAPIPE_MODEL_PATH=/app/.cache/mediapipe
ENV ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json
ENV TMPDIR=/app/.cache

# -------- System Dependencies --------
# NGC image berbasis Ubuntu. Kita install library tambahan yang dibutuhkan CV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    libhdf5-dev libqt5core5a libqt5gui5 libqt5widgets5 libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Prepare Cache Directories --------
# Buat folder cache dulu agar permission bisa di-set nanti
RUN mkdir -p /app/.cache/mediapipe /app/.cache/ultralytics

# -------- Python Packages --------
COPY requirements.txt .
# Upgrade pip bawaan NGC image
RUN pip install --upgrade pip wheel setuptools

# Install dependencies.
# Note: Image NGC sudah include PyTorch & TorchVision yang dioptimasi untuk ARM64.
# Pastikan requirements.txt TIDAK me-reinstall torch/torchvision agar tidak conflict.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    opencv-python-headless \
    onnxruntime-gpu \
    ultralytics \
    websockets \
    fastapi \
    uvicorn[standard]

# -------- Copy Project --------
COPY . .

# -------- OpenShift Permission Fix (CRITICAL) --------
# OpenShift menjalankan container dengan UID acak, tapi user tersebut
# PASTI masuk dalam Group 0 (root).
# Kita beri izin Group 0 (root) untuk read/write di folder /app.
# Ini menggantikan trik 'appuser' yang sering gagal di OpenShift.
RUN chgrp -R 0 /app && \
    chmod -R g+rwX /app

# -------- Optional: Pre-download YOLO models --------
# Karena permission sudah dibenerin di langkah atas, script ini aman jalan.
RUN python3 - <<'PY'
try:
    from ultralytics import YOLO
    # Download model ke folder cache yang sudah kita chmod tadi
    for m in ['yolo11m.pt', 'yolov8n.pt', 'yolov8m.pt']:
        YOLO(m)
except Exception as e:
    print(f"⚠️  Model download skipped: {e}")
PY

# -------- Switch User --------
# Di OpenShift, USER instruction ini sebenarnya akan di-override oleh random UID.
# Tapi kita set ke numeric ID acak (misal 1001) sebagai best practice 
# untuk menandakan image ini tidak butuh root.
USER 1001

# -------- Port & Health Check --------
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# -------- Default Command --------
CMD ["python", "server/server2.py"]
