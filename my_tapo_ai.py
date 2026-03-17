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
from dotenv import load_dotenv
from ultralytics import YOLO

# Load credentials from .env file
load_dotenv()

# Go2RTC Server Configuration
GO2RTC_IP = os.getenv("GO2RTC_IP", "192.168.8.8")
GO2RTC_PORT = os.getenv("GO2RTC_PORT", "8554")

CAMERAS = [
    {"ip": "192.168.8.102", "name": "MainView", "go2rtc_stream": "UPatio"},
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
        cap = cv2.VideoCapture(self.rtsp_url)
        while True:
            if not cap.isOpened():
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1)
                continue
            
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
                with open(f"{current_file}.json", 'w') as f:
                    json.dump(meta, f)
                print(f"[{self.name}] Saved event to {current_file}")

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
                    self.start_recording((w, h))

            self.last_frame = frame

            # Only append to pre-roll when we are NOT currently recording
            if not target_detected and not self.is_recording:
                self.buffer.append(frame.copy())

            # Offload frame save to IO thread
            if self.is_recording:
                self.write_queue.put(("FRAME", frame.copy()))
                if time.time() - self.last_detection_time > COOLDOWN_PERIOD:
                    self.stop_recording()

    def start_recording(self, size):
        self.is_recording = True
        self.start_time = time.time()
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
            "cam": self.name, 
            "objs": list(self.event_metadata["objects"]), 
            "conf": round(self.event_metadata["max_conf"], 2)
        }
        self.write_queue.put(("STOP", meta))

if __name__ == "__main__":
    if not os.path.exists(RECORDING_PATH): os.makedirs(RECORDING_PATH)
    trackers = [SmartTracker(c) for c in CAMERAS]
    for t in trackers: t.start()
    
    # Placeholder black frame for cameras that taking longer to start
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    while True:
        frames = []
        for t in trackers:
            if t.last_frame is not None:
                frames.append(cv2.resize(t.last_frame, (640, 360)))
            else:
                frames.append(blank_frame)
                
        # Now UI will load instantly, even if a camera is offline
        cv2.imshow("M4 Tapo Dashboard", np.vstack(frames))
        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break