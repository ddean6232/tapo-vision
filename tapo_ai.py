import cv2
import time
import os
from ultralytics import YOLO

# --- CONFIG ---
model = YOLO('yolov8n.pt') 
RTSP_URL = "rtsp://192.168.8.8:8554/MainView"
SAVE_DIR = "captures"
COOLDOWN_SECONDS = 10  # Wait this long after person leaves to close file

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

cap = cv2.VideoCapture(RTSP_URL)
recording = False
out = None
last_detection_time = 0

print("Monitoring for people... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Only look for 'person' (Class 0)
    results = model(frame, stream=True, conf=0.5, classes=[0]) 
    
    person_detected = False
    for r in results:
        if len(r.boxes) > 0:
            person_detected = True
            last_detection_time = time.time()
            frame = r.plot() # Annotate the frame

    # Logic: Start or Continue Recording
    if person_detected:
        if not recording:
            print("Person detected! Recording started.")
            recording = True
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{SAVE_DIR}/person_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w, _ = frame.shape
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        
        out.write(frame)

    # Logic: Stop Recording after cooldown
    elif recording:
        out.write(frame) # Keep recording for the cooldown period
        if time.time() - last_detection_time > COOLDOWN_SECONDS:
            print("Cooldown finished. Saving file.")
            recording = False
            out.release()

    cv2.imshow("Maya Beach Security Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
if out: out.release()
cv2.destroyAllWindows()
