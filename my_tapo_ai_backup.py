import os
# Force YOLO offline mode before importing ultralytics (prevents update checks / telemetry)
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB"] = "false"

import cv2
import threading
import time
import json
import queue
import numpy as np
from collections import deque
import logging
from dotenv import load_dotenv
from ultralytics import YOLO

# Load credentials from .env file
load_dotenv()

# Configure Logging
LOG_FILE = "tapo_ai.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TapoAI")

# Go2RTC Server Configuration
GO2RTC_IP = os.getenv("GO2RTC_IP", "192.168.8.8")
GO2RTC_PORT = os.getenv("GO2RTC_PORT", "8554")

CAMERAS = [
    {"ip": "192.168.8.102", "name": "UPatio", "go2rtc_stream": "UPatio"},
    {"ip": "192.168.8.103", "name": "Inside", "go2rtc_stream": "Inside"},
    {"ip": "192.168.8.104", "name": "LPatio", "go2rtc_stream": "LPatio"},
]

RECORDING_PATH = "./recordings/"
COOLDOWN_PERIOD = 8   
PRE_ROLL_SECONDS = 5
FPS = 20  

# Load YOLOv8 and move to M4 GPU. 
# Added a Lock to make model thread-safe across multiple streams.
model = YOLO('yolov8n.pt').to('mps')
model_lock = threading.Lock()

class SmartTracker:
    master_log_lock = threading.Lock()

    def __init__(self, cam_info):
        self.ip = cam_info['ip']
        self.name = cam_info['name']
        self.go2rtc_stream = cam_info['go2rtc_stream']
        self.last_frame = None
        self.latest_rtsp_frame = None  # Continuously updated by reader thread
        
        # Using go2rtc for the video stream instead of connecting to the camera directly!
        self.rtsp_url = f"rtsp://{GO2RTC_IP}:{GO2RTC_PORT}/{self.go2rtc_stream}"
        self.buffer = deque(maxlen=PRE_ROLL_SECONDS * FPS)
        self.is_recording = False
        self.write_queue = queue.Queue()  # For non-blocking video writing
        self.last_detection_time = 0
        self.event_metadata = {"objects": set(), "max_conf": 0.0}

    def start(self):
        # Start the background workers
        threading.Thread(target=self._writer_worker, daemon=True).start()
        threading.Thread(target=self._rtsp_reader, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    def _rtsp_reader(self):
        """Continuously pulls frames from RTSP to ensure frame is fresh (near 0 lag)."""
        # Force TCP transport via URL option — more reliable than UDP on local networks
        rtsp_url_tcp = self.rtsp_url + "?rtsp_transport=tcp"

        while True:
            print(f"[{self.name}] Connecting to stream...")
            cap = cv2.VideoCapture(rtsp_url_tcp, cv2.CAP_FFMPEG)

            # Fail fast: don't let FFmpeg hang for 10+ minutes on a dead stream
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5s to open
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5s per frame read
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)              # Keep only latest frame

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

                # Update latest frame immediately for AI thread to grab
                self.latest_rtsp_frame = frame

    def _writer_worker(self):
        """Dedicated thread to handle disk I/O for video writing."""
        writer = None
        current_file = None
        while True:
            task = self.write_queue.get()
            if task is None: continue
                
            task_type, payload = task
            if task_type == "START":
                filepath, fps, size = payload
                current_file = filepath
                writer = cv2.VideoWriter(f"{filepath}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
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
                            with open(master_log_path, 'r') as f:
                                log_data = json.load(f)
                        except Exception:
                            pass
                    
                    log_data.append(meta)
                    
                    with open(master_log_path, 'w') as f:
                        json.dump(log_data, f, indent=4)
                        
                logger.info(f"[{self.name}] Saved video {current_file}.mp4 and updated unified JSON event log")

    def _run(self):
        last_processed_frame = None
        last_inference_time = 0
        last_boxes = []

        while True:
            # Grab latest real-time frame from reader thread
            frame = self.latest_rtsp_frame
            if frame is None or frame is last_processed_frame:
                time.sleep(0.02)
                continue

            last_processed_frame = frame
            frame = frame.copy() # Local copy for annotation
            h, w = frame.shape[:2]

            cv2.putText(frame, f"{self.name} | {time.strftime('%H:%M:%S')}", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # --- AI Inference ---
            # Using lock to prevent MPS crash from overlapping predictions
            # Throttled to max 5 times a second to save GPU, but running at full resolution and 65% confidence
            now = time.time()
            if now - last_inference_time > 0.2:  # Throttle AI to max 5 times a second
                last_inference_time = now
                with model_lock:
                    results = model.predict(frame, classes=[0, 2, 7], conf=0.65, verbose=False)
                last_boxes = results[0].boxes
            
            target_detected = len(last_boxes) > 0
            
            if target_detected:
                if now - last_inference_time <= 0.2: # Only update detection time when an AI check actually happened
                    self.last_detection_time = time.time()
                
                for box in last_boxes:
                    c = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.cpu().item())
                    label = model.names[int(box.cls)]
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
                    self.event_metadata["objects"].add(label)
                    self.event_metadata["max_conf"] = max(self.event_metadata["max_conf"], conf)

                if not self.is_recording:
                    detected_labels = ", ".join(list(self.event_metadata["objects"])).title() if self.event_metadata["objects"] else "Activity"
                    self.start_recording((w, h), trigger_items=detected_labels)

            self.last_frame = frame

            # Only append to pre-roll when we are NOT currently recording
            if not target_detected and not self.is_recording:
                self.buffer.append(frame.copy())

            # Offload frame save to IO thread
            if self.is_recording:
                self.write_queue.put(("FRAME", frame.copy()))
                if time.time() - self.last_detection_time > COOLDOWN_PERIOD:
                    self.stop_recording()

    def start_recording(self, size, trigger_items="Activity"):
        self.is_recording = True
        self.start_time = time.time()
        
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
        
        path = f"{RECORDING_PATH}{self.name}_{time.strftime('%m%d-%H%M%S')}"
        
        # Dispatch command to writer thread
        self.write_queue.put(("START", (path, FPS, size)))
        
        # Unload the entire pre-roll buffer concurrently
        for f in self.buffer:
            self.write_queue.put(("FRAME", f))
        
        self.buffer.clear()

    def stop_recording(self):
        self.is_recording = False
        meta = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "cam": self.name, 
            "objs": list(self.event_metadata["objects"]), 
            "conf": round(self.event_metadata["max_conf"], 2)
        }
        self.write_queue.put(("STOP", meta))

if __name__ == "__main__":
    if not os.path.exists(RECORDING_PATH): os.makedirs(RECORDING_PATH)
    
    trackers = [SmartTracker(c) for c in CAMERAS]
    for t in trackers: t.start()

    from flask import Flask, Response, render_template_string
    import rumps
    import webbrowser

    flask_app = Flask(__name__)
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def generate_frames():
        while True:
            frames = []
            for t in trackers:
                if t.last_frame is not None:
                    frames.append(cv2.resize(t.last_frame, (640, 360)))
                else:
                    frames.append(blank_frame)

            # Create Composite Dashboard
            composite = np.vstack(frames)
            ret, buffer = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)

    @flask_app.route('/')
    def index():
        return '''
        <html>
            <head>
                <title>Tapo-Vision Dashboard</title>
                <style>
                    body { background-color: #0b0c10; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                    img { max-height: 96vh; max-width: 96vw; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
                </style>
            </head>
            <body>
                <img src="/video_feed" />
            </body>
        </html>
        '''

    @flask_app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    class TapoMenubarApp(rumps.App):
        def __init__(self):
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "menubar_icon.png")
            super(TapoMenubarApp, self).__init__("", icon=icon_path, template=True, quit_button=None)
            self.menu = [
                rumps.MenuItem("View Live Dashboard", callback=self.open_dashboard),
                rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
                None,
                rumps.MenuItem("Quit Tapo-Vision", callback=self.quit_app)
            ]

        def open_dashboard(self, _):
            webbrowser.open("http://127.0.0.1:5050")

        def open_recordings(self, _):
            import subprocess
            subprocess.run(["open", os.path.abspath(RECORDING_PATH)])

        def quit_app(self, _):
            logger.info("Shutdown requested via Menubar.")
            rumps.quit_application()

    # Start Flask Server in background
    threading.Thread(target=lambda: flask_app.run(host='127.0.0.1', port=5050, debug=False, use_reloader=False), daemon=True).start()

    logger.info("Starting Mac Menubar App. Check your menubar (top right) for the 📹 icon.")
    # Start Menubar App (blocking)
    TapoMenubarApp().run()