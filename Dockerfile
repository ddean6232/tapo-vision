FROM python:3.11-slim

# System deps for OpenCV headless + OpenVINO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the lightning-fast 'uv' installer directly from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Use uv to install packages into the system python. 
# We explicitly force the CPU index so it NEVER downloads Nvidia/CUDA bloat.
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    ultralytics \
    opencv-python-headless \
    python-dotenv \
    numpy \
    openvino \
    flask \
    imageio[ffmpeg]

# Copy application code and model
COPY tapo_docker.py .
COPY dashboard.py .
COPY yolov8n.pt .

# Create recordings directory
RUN mkdir -p /app/recordings

# Unbuffered output so Docker logs work in real time
ENV PYTHONUNBUFFERED=1
ENV YOLO_OFFLINE=1
ENV ULTRALYTICS_HUB=false

CMD ["python", "tapo_docker.py"]
