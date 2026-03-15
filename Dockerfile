FROM python:3.11-slim

# System deps for OpenCV headless + OpenVINO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
# opencv-python-headless = no GUI deps (smaller image)
# openvino = Intel CPU optimization for YOLO (2-3x speedup on X220)
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    pytapo \
    python-dotenv \
    numpy \
    openvino

# Copy application code and model
COPY tapo_docker.py .
COPY yolov8n.pt .

# Create recordings directory
RUN mkdir -p /app/recordings

# Unbuffered output so Docker logs work in real time
ENV PYTHONUNBUFFERED=1
ENV YOLO_OFFLINE=1
ENV ULTRALYTICS_HUB=false

CMD ["python", "tapo_docker.py"]
