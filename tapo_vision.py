"""
tapo_vision.py — Headless, CPU-optimized build for Docker / CHEETAH server deployment.
Based on my_tapo_ai.py but tuned for low-power Intel hardware (2C/4T Sandy Bridge).

Key differences from my_tapo_ai.py:
  - CPU-only (no MPS/GPU)
  - OpenVINO auto-export for 2-3x Intel speedup
  - Headless (no cv2.imshow)
  - Lower inference rate (2 Hz) and smaller input (416px)
  - Memory-safe write queue (capped at 500 frames)
  -  RTSP reconnect with exponential backoff
  - Graceful shutdown via SIGINT/SIGTERM
  - Structured logging instead of print()
"""

import os
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB"] = "false"

import cv2
import threading
import time
import json
import queue
import signal
import sys
import logging
import numpy as np
import imageio
from collections import deque
from dotenv import load_dotenv
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tapo-vision")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()

GO2RTC_IP   = os.getenv("GO2RTC_IP", "192.168.8.8")
GO2RTC_PORT = os.getenv("GO2RTC_PORT", "8554")

CAMERAS = [
    {"ip": "192.168.8.102", "name": "UPatio", "go2rtc_stream": "UPatio"},
    {"ip": "192.168.8.103", "name": "LPatio",   "go2rtc_stream": "LPatio"},
    {"ip": "192.168.8.104", "name": "Inside",   "go2rtc_stream": "Inside"},
]

RECORDING_PATH    = os.getenv("RECORDING_PATH", "./recordings/")
COOLDOWN_PERIOD   = 8
PRE_ROLL_SECONDS  = 5
FPS               = int(os.getenv("CAMERA_FPS", "20")) # Match your camera's actual framerate to fix slow-motion
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.5"))
INPUT_SIZE         = 416        # Smaller input = faster inference (was 640)
CONFIDENCE         = 0.60       # Slightly lower to compensate for smaller input
MAX_WRITE_QUEUE    = 500        # Cap to prevent OOM (~500 frames ≈ 300 MB)

# ──────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────
shutdown_event = threading.Event()

def _signal_handler(sig, frame):
    log.info("Shutdown signal received, stopping gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ──────────────────────────────────────────────
# Model loading — Apple Silicon GPU (MPS)
# ──────────────────────────────────────────────
log.info("Loading YOLOv8n on Apple Silicon GPU (MPS)...")
model = YOLO("yolov8n.pt")
model.to("mps")  # Force use of the M4 Metal GPU

model_lock = threading.Lock()


class SmartTracker:
    def __init__(self, cam_info):
        self.ip   = cam_info["ip"]
        self.name = cam_info["name"]
        self.go2rtc_stream = cam_info["go2rtc_stream"]
        self.latest_rtsp_frame = None

        # RTSP via go2rtc
        self.rtsp_url = f"rtsp://{GO2RTC_IP}:{GO2RTC_PORT}/{self.go2rtc_stream}"

        self.buffer = deque(maxlen=PRE_ROLL_SECONDS * FPS)
        self.is_recording = False
        self.write_queue = queue.Queue(maxsize=MAX_WRITE_QUEUE)
        self.last_detection_time = 0
        self.event_metadata = {"objects": set(), "max_conf": 0.0}

    # ── Thread entry points ──────────────────

    def start(self):
        threading.Thread(target=self._writer_worker, name=f"{self.name}-writer", daemon=True).start()
        threading.Thread(target=self._rtsp_reader,   name=f"{self.name}-rtsp",   daemon=True).start()
        threading.Thread(target=self._run,           name=f"{self.name}-ai",     daemon=True).start()
        log.info(f"[{self.name}] All threads started")

    def _rtsp_reader(self):
        """Pull frames from go2rtc with exponential backoff on failure."""
        backoff = 1
        cap = None

        while not shutdown_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    log.info(f"[{self.name}] Connecting to RTSP: {self.rtsp_url}")
                    cap = cv2.VideoCapture(self.rtsp_url)
                    if not cap.isOpened():
                        log.warning(f"[{self.name}] RTSP connect failed, retry in {backoff}s")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                        continue
                    backoff = 1  # Reset on success
                    log.info(f"[{self.name}] RTSP connected")

                ret, frame = cap.read()
                if not ret:
                    log.warning(f"[{self.name}] RTSP read failed, reconnecting in {backoff}s")
                    cap.release()
                    cap = None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

                self.latest_rtsp_frame = frame
                backoff = 1

            except Exception as e:
                log.error(f"[{self.name}] RTSP reader error: {e}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        if cap is not None:
            cap.release()
        log.info(f"[{self.name}] RTSP reader stopped")

    def _writer_worker(self):
        """Disk I/O thread — writes video and metadata."""
        writer = None
        current_file = None

        while not shutdown_event.is_set():
            try:
                task = self.write_queue.get(timeout=1)
            except queue.Empty:
                continue

            if task is None:
                continue

            try:
                task_type, payload = task
                if task_type == "START":
                    filepath, fps, size = payload
                    current_file = filepath
                    # Use imageio to guarantee perfectly encoded, web-compatible H.264 video
                    writer = imageio.get_writer(f"{filepath}.mp4", fps=fps, macro_block_size=None)
                    log.info(f"[{self.name}] Recording started: {filepath}")
                elif task_type == "FRAME":
                    if writer is not None:
                        writer.append_data(cv2.cvtColor(payload, cv2.COLOR_BGR2RGB))
                elif task_type == "STOP":
                    if writer is not None:
                        writer.close()
                        writer = None
                    meta = payload
                    if current_file:
                        with open(f"{current_file}.json", "w") as f:
                            json.dump(meta, f)
                    log.info(f"[{self.name}] Recording saved: {current_file}")
            except Exception as e:
                import traceback
                log.error(f"[{self.name}] Writer error: {traceback.format_exc()}")

        # Cleanup on shutdown
        if writer is not None:
            writer.release()
        log.info(f"[{self.name}] Writer stopped")

    def _run(self):
        """Main AI loop — inference, tracking, recording control."""
        last_processed_frame = None
        last_inference_time = 0
        last_boxes = []
        inference_count = 0

        while not shutdown_event.is_set():
            try:
                frame = self.latest_rtsp_frame
                if frame is None or frame is last_processed_frame:
                    time.sleep(0.01)  # Was 0.05: lowered to prevent dropping frames (light-speed bug)
                    continue

                last_processed_frame = frame
                frame = frame.copy()
                h, w = frame.shape[:2]

                # --- AI Inference (throttled) ---
                now = time.time()
                if now - last_inference_time > INFERENCE_INTERVAL:
                    last_inference_time = now
                    with model_lock:
                        results = model.predict(
                            frame,
                            classes=[0, 2, 7],
                            conf=CONFIDENCE,
                            imgsz=INPUT_SIZE,
                            verbose=False,
                        )
                    last_boxes = results[0].boxes
                    inference_count += 1

                    # Log stats every 100 inferences
                    if inference_count % 100 == 0:
                        log.info(
                            f"[{self.name}] {inference_count} inferences done | "
                            f"recording={self.is_recording} | "
                            f"queue={self.write_queue.qsize()}"
                        )

                target_detected = len(last_boxes) > 0

                if target_detected:
                    if now - last_inference_time <= INFERENCE_INTERVAL:
                        self.last_detection_time = time.time()

                    for box in last_boxes:
                        conf = float(box.conf.cpu().item())
                        label = model.names[int(box.cls)]
                        self.event_metadata["objects"].add(label)
                        self.event_metadata["max_conf"] = max(
                            self.event_metadata["max_conf"], conf
                        )

                    if not self.is_recording:
                        self.start_recording((w, h))

                # Pre-roll buffer (only when not recording)
                if not target_detected and not self.is_recording:
                    self.buffer.append(frame.copy())

                # Write frames & check cooldown
                if self.is_recording:
                    try:
                        self.write_queue.put_nowait(("FRAME", frame.copy()))
                    except queue.Full:
                        log.warning(f"[{self.name}] Write queue full, dropping frame")
                    if time.time() - self.last_detection_time > COOLDOWN_PERIOD:
                        self.stop_recording()

            except Exception as e:
                log.error(f"[{self.name}] AI loop error: {e}")
                time.sleep(1)

        log.info(f"[{self.name}] AI loop stopped")

    def start_recording(self, size):
        self.is_recording = True
        self.start_time = time.time()
        self.event_metadata = {"objects": set(), "max_conf": 0.0}

        path = f"{RECORDING_PATH}{self.name}_{time.strftime('%m%d-%H%M%S')}"
        self.write_queue.put(("START", (path, FPS, size)))

        for f in self.buffer:
            try:
                self.write_queue.put_nowait(("FRAME", f))
            except queue.Full:
                break
        self.buffer.clear()

    def stop_recording(self):
        self.is_recording = False
        meta = {
            "cam": self.name,
            "objs": list(self.event_metadata["objects"]),
            "conf": round(self.event_metadata["max_conf"], 2),
        }
        self.write_queue.put(("STOP", meta))


# ──────────────────────────────────────────────
# Main — headless, no GUI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RECORDING_PATH, exist_ok=True)

    log.info("=" * 50)
    log.info("Tapo Vision — Bare Metal / MPS Edition")
    log.info(f"Cameras: {len(CAMERAS)}")
    log.info(f"Inference: {INPUT_SIZE}px @ {1/INFERENCE_INTERVAL:.0f} Hz, conf={CONFIDENCE}")
    log.info(f"Recordings: {RECORDING_PATH}")
    log.info("=" * 50)

    trackers = [SmartTracker(c) for c in CAMERAS]
    for t in trackers:
        t.start()

    # ──────────────────────────────────────────────
    # Launch Dashboard seamlessly in the background
    # ──────────────────────────────────────────────
    try:
        import dashboard
        import waitress
        # Use Waitress (production WSGI server) instead of Flask dev server to prevent "Too many open files" leaks
        threading.Thread(
            target=lambda: waitress.serve(dashboard.app, host="0.0.0.0", port=38180, clear_untrusted_proxy_headers=True),
            name="Dashboard",
            daemon=True
        ).start()
        log.info("Dashboard web server started on port 38180 (Waitress)")
    except Exception as e:
        log.error(f"Failed to start dashboard: {e}")

    # ──────────────────────────────────────────────
    # Launch Offline Analyzer in the background
    # ──────────────────────────────────────────────
    try:
        import offline_analyzer
        threading.Thread(
            target=offline_analyzer.start_analyzer_loop,
            name="AnalyzerLoop",
            daemon=True
        ).start()
        log.info("Offline Analyzer scheduled (running every 5 minutes in the background)")
    except Exception as e:
        log.error(f"Failed to start offline analyzer: {e}")

    log.info("All cameras running. Press Ctrl+C to stop.")

    # Keep main thread alive, log heartbeat every 60s
    heartbeat = 0
    while not shutdown_event.is_set():
        shutdown_event.wait(timeout=60)
        heartbeat += 1
        if not shutdown_event.is_set():
            status = []
            for t in trackers:
                has_frame = "live" if t.latest_rtsp_frame is not None else "no-signal"
                rec = "REC" if t.is_recording else "idle"
                status.append(f"{t.name}={has_frame}/{rec}")
            log.info(f"[Heartbeat #{heartbeat}] {' | '.join(status)}")

    log.info("Shutdown complete.")
