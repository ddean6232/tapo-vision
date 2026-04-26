# 📹 Tapo-Vision

AI-powered multi-camera security surveillance system for Tapo PTZ cameras. Optimized for Apple M4 GPU (local) and Intel CPU (Docker/OpenVINO).

## Features

- **Real-time Detection**: Uses YOLOv8 Nano to detect people, cars, and trucks.
- **Smart Recording**: 5-second pre-roll buffer + event-based recording with JSON metadata.
- **M4 GPU Optimized**: High-performance inference using Apple Metal (MPS).
- **Docker Ready**: Headless version with OpenVINO optimization for continuous 24/7 monitoring on low-power hardware (like ThinkPad X220 or N100 mini PCs).
- **Go2RTC Integration**: High-speed, low-latency RTSP streaming via go2rtc.

## Repo Structure

- `my_tapo_ai.py`: Main dashboard application with GUI (best for MacBook Pro M4).
- `tapo_vision.py`: Headless, CPU-optimized version (best for 24/7 Docker deployment).
- `Dockerfile` / `docker-compose.yml`: Deployment configs for Docker.

## Setup

1. Copy `.env.example` to `.env` and fill in your camera credentials.
2. Install dependencies: `pip install ultralytics python-dotenv opencv-python`.
3. Run `python my_tapo_ai.py`.

## Optimization Notes

For Intel CPU deployment, the system automatically exports the YOLO model to **OpenVINO** format on the first run, providing a 2-3x speedup on older hardware like the ThinkPad X220.
