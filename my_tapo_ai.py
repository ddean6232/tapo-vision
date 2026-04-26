#!/usr/bin/env python3
"""
Tapo Vision — Main Camera Monitor
Usage:
    python my_tapo_ai.py
    python my_tapo_ai.py --recordings /Volumes/NAS/tapo/
    python my_tapo_ai.py --go2rtc-ip 192.168.1.5 --go2rtc-port 8554
    python my_tapo_ai.py --port 5050 --log-level DEBUG
"""
import os
# Force YOLO offline mode before importing ultralytics (prevents update checks / telemetry)
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB"] = "false"

import sys
import cv2
import threading
import time
import json
import queue
import signal
import argparse
import logging
import numpy as np
from collections import deque
from ultralytics import YOLO

# ── CLI Arguments ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tapo Vision — AI-powered security camera monitor with macOS menu bar integration."
    )
    parser.add_argument(
        "--recordings",
        default="./recordings/",
        help="Path to the folder where recordings and the event log are saved. Default: ./recordings/"
    )
    parser.add_argument(
        "--go2rtc-ip",
        default="192.168.8.8",
        help="IP address of the go2rtc server. Default: 192.168.8.8"
    )
    parser.add_argument(
        "--go2rtc-port",
        default="8554",
        help="RTSP port of the go2rtc server. Default: 8554"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port for the local live-view web dashboard. Default: 5050"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO"
    )
    return parser.parse_args()

# ── Parse args early so config is available at module level ────────────────────
args = parse_args()
RECORDING_PATH = os.path.abspath(args.recordings)
GO2RTC_IP      = args.go2rtc_ip
GO2RTC_PORT    = args.go2rtc_port
DASHBOARD_PORT = args.port

os.makedirs(RECORDING_PATH, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RECORDING_PATH, "tapo_ai.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("TapoAI")

# ── Graceful shutdown ───────────────────────────────────────────────────────────
def _handle_signal(signum, frame):
    logger.info(f"🛑 Signal {signum} received. Shutting down Tapo Vision...")
    os._exit(0)  # Hard exit — kills all daemon threads cleanly

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Camera configuration ────────────────────────────────────────────────────────
CAMERAS = [
    {"ip": "192.168.8.102", "name": "UPatio", "go2rtc_stream": "UPatio"},
    {"ip": "192.168.8.103", "name": "Inside",  "go2rtc_stream": "Inside"},
    {"ip": "192.168.8.104", "name": "LPatio", "go2rtc_stream": "LPatio"},
]

COOLDOWN_PERIOD  = 8    # seconds after last detection before stopping recording
PRE_ROLL_SECONDS = 5    # seconds of buffer to prepend to each recording
FPS              = 20

# ── Load YOLO model ─────────────────────────────────────────────────────────────
logger.info("Loading YOLO model on MPS (Apple Silicon GPU)...")
model      = YOLO("yolov8n.pt").to("mps")
model_lock = threading.Lock()
logger.info("YOLO model ready.")

# ── SmartTracker ────────────────────────────────────────────────────────────────
class SmartTracker:
    master_log_lock = threading.Lock()

    def __init__(self, cam_info):
        self.ip               = cam_info["ip"]
        self.name             = cam_info["name"]
        self.go2rtc_stream    = cam_info["go2rtc_stream"]
        self.last_frame       = None
        self.latest_rtsp_frame = None

        self.rtsp_url         = f"rtsp://{GO2RTC_IP}:{GO2RTC_PORT}/{self.go2rtc_stream}"
        self.buffer           = deque(maxlen=PRE_ROLL_SECONDS * FPS)
        self.is_recording     = False
        self.write_queue      = queue.Queue()
        self.last_detection_time = 0
        self.event_metadata   = {"objects": set(), "max_conf": 0.0}

    def start(self):
        threading.Thread(target=self._writer_worker, daemon=True).start()
        threading.Thread(target=self._rtsp_reader,   daemon=True).start()
        threading.Thread(target=self._run,            daemon=True).start()

    def _rtsp_reader(self):
        """Continuously pulls frames from RTSP to ensure near-zero lag."""
        rtsp_url_tcp = self.rtsp_url + "?rtsp_transport=tcp"
        while True:
            logger.info(f"[{self.name}] Connecting to stream: {rtsp_url_tcp}")
            cap = cv2.VideoCapture(rtsp_url_tcp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logger.warning(f"[{self.name}] Stream unavailable, retrying in 5s...")
                cap.release()
                time.sleep(5)
                continue

            logger.info(f"[{self.name}] Stream connected.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{self.name}] Stream lost, reconnecting in 3s...")
                    cap.release()
                    time.sleep(3)
                    break
                self.latest_rtsp_frame = frame

    def _writer_worker(self):
        """Dedicated IO thread — handles all disk writes without blocking the AI loop."""
        writer       = None
        current_file = None
        while True:
            task = self.write_queue.get()
            if task is None:
                continue

            task_type, payload = task
            if task_type == "START":
                filepath, fps, size = payload
                current_file = filepath
                writer = cv2.VideoWriter(
                    f"{filepath}.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, size
                )
            elif task_type == "FRAME":
                if writer is not None:
                    writer.write(payload)
            elif task_type == "STOP":
                if writer is not None:
                    writer.release()
                    writer = None

                meta = payload
                meta["video_file"] = f"{os.path.basename(current_file)}.mp4"
                master_log_path = os.path.join(RECORDING_PATH, "unified_events_log.json")

                with self.master_log_lock:
                    log_data = []
                    if os.path.exists(master_log_path):
                        try:
                            with open(master_log_path, "r") as f:
                                log_data = json.load(f)
                        except Exception:
                            pass
                    log_data.append(meta)
                    with open(master_log_path, "w") as f:
                        json.dump(log_data, f, indent=4)

                logger.info(f"[{self.name}] Saved {current_file}.mp4 and updated event log.")

    def _run(self):
        last_processed_frame = None
        last_inference_time  = 0
        last_boxes           = []

        while True:
            frame = self.latest_rtsp_frame
            if frame is None or frame is last_processed_frame:
                time.sleep(0.02)
                continue

            last_processed_frame = frame
            frame = frame.copy()
            h, w = frame.shape[:2]

            cv2.putText(frame, f"{self.name} | {time.strftime('%H:%M:%S')}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            now = time.time()
            if now - last_inference_time > 0.2:  # ~5 FPS AI inference
                last_inference_time = now
                with model_lock:
                    results = model.predict(frame, classes=[0, 2, 7], conf=0.65, verbose=False)
                last_boxes = results[0].boxes

            target_detected = len(last_boxes) > 0

            if target_detected:
                if now - last_inference_time <= 0.2:
                    self.last_detection_time = time.time()

                for box in last_boxes:
                    c    = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.cpu().item())
                    label = model.names[int(box.cls)]
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
                    self.event_metadata["objects"].add(label)
                    self.event_metadata["max_conf"] = max(self.event_metadata["max_conf"], conf)

                if not self.is_recording:
                    detected_labels = (
                        ", ".join(self.event_metadata["objects"]).title()
                        if self.event_metadata["objects"] else "Activity"
                    )
                    self.start_recording((w, h), trigger_items=detected_labels)

            self.last_frame = frame

            if not target_detected and not self.is_recording:
                self.buffer.append(frame.copy())

            if self.is_recording:
                self.write_queue.put(("FRAME", frame.copy()))
                if time.time() - self.last_detection_time > COOLDOWN_PERIOD:
                    self.stop_recording()

    def start_recording(self, size, trigger_items="Activity"):
        self.is_recording = True
        self.start_time   = time.time()

        try:
            import rumps
            rumps.notification(
                title=f"🚨 Tapo-Vision: {self.name}",
                subtitle=f"{trigger_items} Detected",
                message="Camera event recording started."
            )
        except Exception:
            pass

        self.event_metadata = {"objects": set(), "max_conf": 0.0}
        path = os.path.join(RECORDING_PATH, f"{self.name}_{time.strftime('%m%d-%H%M%S')}")
        self.write_queue.put(("START", (path, FPS, size)))
        for f in self.buffer:
            self.write_queue.put(("FRAME", f))
        self.buffer.clear()

    def stop_recording(self):
        self.is_recording = False
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cam":       self.name,
            "objs":      list(self.event_metadata["objects"]),
            "conf":      round(self.event_metadata["max_conf"], 2),
        }
        self.write_queue.put(("STOP", meta))


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"📁 Recordings path : {RECORDING_PATH}")
    logger.info(f"🌐 go2rtc server   : {GO2RTC_IP}:{GO2RTC_PORT}")
    logger.info(f"📺 Dashboard       : http://127.0.0.1:{DASHBOARD_PORT}")

    trackers = [SmartTracker(c) for c in CAMERAS]
    for t in trackers:
        t.start()

    from flask import Flask, Response
    import rumps
    import webbrowser

    flask_app   = Flask(__name__)
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def generate_frames():
        while True:
            frames = [
                cv2.resize(t.last_frame, (640, 360)) if t.last_frame is not None else blank_frame
                for t in trackers
            ]
            composite = np.vstack(frames)
            ret, buffer = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.05)

    @flask_app.route("/")
    def index():
        return """
        <html>
            <head>
                <title>Tapo-Vision Dashboard</title>
                <style>
                    body { background-color: #0b0c10; display: flex; justify-content: center;
                           align-items: center; height: 100vh; margin: 0; }
                    img  { max-height: 96vh; max-width: 96vw; border-radius: 12px;
                           box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
                </style>
            </head>
            <body><img src="/video_feed" /></body>
        </html>
        """

    @flask_app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    class TapoMenubarApp(rumps.App):
        def __init__(self):
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "menubar_icon.png")
            super().__init__("", icon=icon_path, template=True, quit_button=None)
            self.menu = [
                rumps.MenuItem("View Live Dashboard",    callback=self.open_dashboard),
                rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
                None,
                rumps.MenuItem("Quit Tapo-Vision", callback=self.quit_app),
            ]

        def open_dashboard(self, _):
            webbrowser.open(f"http://127.0.0.1:{DASHBOARD_PORT}")

        def open_recordings(self, _):
            import subprocess
            subprocess.run(["open", RECORDING_PATH])

        def quit_app(self, _):
            """Fully terminates the process including all background threads."""
            logger.info("Shutdown requested via Menubar — exiting cleanly.")
            os._exit(0)  # Hard exit: kills all daemon threads, no zombie processes

    threading.Thread(
        target=lambda: flask_app.run(
            host="127.0.0.1", port=DASHBOARD_PORT, debug=False, use_reloader=False
        ),
        daemon=True
    ).start()

    logger.info("Starting Mac Menubar App. Check your menubar for the 📹 icon.")
    TapoMenubarApp().run()