# Use NVIDIA PyTorch base image with CUDA support for H200
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5test5 \
    libgstreamer-plugins-base1.0-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies needed for GPU acceleration and MediaPipe
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    mediapipe \
    mediapipe-cpu \
    onnxruntime-gpu \
    onnxruntime-tools

# Copy the application code
COPY . .

# Download models if they don't exist (optional - you can mount them as volumes)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" 2>/dev/null || echo "Model download skipped"
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || echo "Model download skipped"
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" 2>/dev/null || echo "Model download skipped"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "server/production_server.py"]