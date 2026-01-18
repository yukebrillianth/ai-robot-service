FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    MEDIAPIPE_MODEL_PATH=/app/.cache/mediapipe \
    ULTRALYTICS_SETTINGS=/app/.cache/ultralytics/settings.json \
    TMPDIR=/app/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless ultralytics websockets fastapi "uvicorn[standard]" "numpy<2.0" && \
    rm -rf /root/.cache/pip

COPY . .

RUN chgrp -R 0 /app && chmod -R g+rwX /app && \
    mkdir -p /app/.cache/mediapipe /app/.cache/ultralytics

RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

USER 1001
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server/server2.py"]