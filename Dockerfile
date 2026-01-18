FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json \
    TMPDIR=/app/.cache

# Instal system dependencies yang diperlukan OpenCV & Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Buat folder cache di awal agar permission benar
RUN mkdir -p /app/.cache/matplotlib /app/.cache/ultralytics /app/.cache/mediapipe

COPY requirements.txt .

# Instalasi Python packages
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 && \
    rm -rf /root/.cache/pip

COPY . .

# Fix permission untuk OpenShift
RUN chgrp -R 0 /app && chmod -R g+rwX /app

# Download model saat build time
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# USER 1001 sesuai standar keamanan OpenShift
USER 1001
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server/server2.py"]