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
from pytapo import Tapo

# Load credentials from .env file
load_dotenv()

# --- CONFIGURATION ---
USER_ADMIN = os.getenv("USER_ADMIN")
PASS_ADMIN = os.getenv("PASS_ADMIN")
USER_RTSP = os.getenv("USER_RTSP")
PASS_RTSP = os.getenv("PASS_RTSP")

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
        self.privacy_active = False
        
        self.last_motor_time = 0
        self.move_cooldown = 0.6  
        
        try:
            self.tapo = Tapo(self.ip, USER_ADMIN, PASS_ADMIN)
            self.check_privacy()
            self.go_home()
        except Exception as e:
            print(f"[{self.name}] Failed to init PyTapo: {e}")
            self.tapo = None

        # Using go2rtc for the video stream instead of connecting to the camera directly!
        self.rtsp_url = f"rtsp://{GO2RTC_IP}:{GO2RTC_PORT}/{self.go2rtc_stream}"
        
        self.buffer = deque(maxlen=PRE_ROLL_SECONDS * FPS)
        self.is_recording = False
        self.write_queue = queue.Queue()  # For non-blocking video writing
        self.last_detection_time = 0
        self.event_metadata = {"objects": set(), "max_conf": 0.0}

    def check_privacy(self):
        try:
            info = self.tapo.getPrivacyMode()
            self.privacy_active = (info == "on" or info.get("enabled") == "on" or info.get("enabled") is True)
        except Exception:
            self.privacy_active = False

    def smooth_move(self, error_x, frame_width, label="person"):
        """Only nudge the camera when the subject drifts into the outer edges of the frame.
        Vehicles get a faster response since they move through the frame quicker."""
        is_vehicle = label in ("car", "truck")
        
        # Vehicles: respond faster (0.3s cooldown), People: stay calm (0.6s cooldown)
        cooldown = 0.3 if is_vehicle else self.move_cooldown
        
        now = time.time()
        if not self.tapo or self.privacy_active or (now - self.last_motor_time < cooldown):
            return

        # Vehicles: tighter deadzone (center 50%), People: wide deadzone (center 70%)
        edge_pct = 0.25 if is_vehicle else 0.35
        edge_threshold = frame_width * edge_pct
        if abs(error_x) < edge_threshold:
            return

        # Vehicles: stronger nudge multiplier & higher cap
        multiplier = 0.04 if is_vehicle else 0.02
        max_step = 8 if is_vehicle else 5
        
        move_val = int(error_x * multiplier)
        move_val = max(-max_step, min(max_step, move_val))
        
        # Minimum nudge of 1 step if we decided to move at all
        if move_val == 0:
            move_val = 1 if error_x > 0 else -1

        try:
            self.last_motor_time = now
            threading.Thread(target=self.tapo.moveMotor, args=(move_val, 0), daemon=True).start()
        except Exception as e:
            print(f"[{self.name}] Motor move failed: {e}")

    def go_home(self):
        if not self.tapo or self.privacy_active: return
        try:
            for method in ['setPreset', 'set_preset']:
                if hasattr(self.tapo, method):
                    getattr(self.tapo, method)(1)
                    break
        except Exception as e:
            # -64303 is MOTOR_BUSY, meaning the motor is already moving or locked
            if "64303" in str(e) or "motor_busy" in str(e).lower():
                pass
            else:
                print(f"[{self.name}] Return to home failed: {e}")

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
        # Privacy Poller
        def poller():
            while True:
                if self.tapo: self.check_privacy()
                time.sleep(10)
        threading.Thread(target=poller, daemon=True).start()
        
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

            if self.privacy_active:
                self.last_frame = frame
                time.sleep(0.05)
                continue

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
                
                top_box = last_boxes[0]
                box_data = top_box.xywh[0].cpu().tolist()
                center_x = box_data[0]
                
                for box in last_boxes:
                    c = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.cpu().item())
                    label = model.names[int(box.cls)]
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
                    self.event_metadata["objects"].add(label)
                    self.event_metadata["max_conf"] = max(self.event_metadata["max_conf"], conf)

                error_x = center_x - (w / 2)
                top_label = model.names[int(top_box.cls)]
                self.smooth_move(error_x, w, label=top_label)

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
        self.go_home()

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