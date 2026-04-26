import os
import time
import json
import cv2
import logging
from google import genai
from google.genai import types
import PIL.Image
from dotenv import load_dotenv

load_dotenv() # Load API keys from .env file
import logging
from pydantic import BaseModel
from typing import List, Optional

class Person(BaseModel):
    clothing: str
    complexion: Optional[str]

class Vehicle(BaseModel):
    make: str
    model: str
    color: str
    type: str

class SceneAnalysis(BaseModel):
    people: Optional[List[Person]]
    vehicles: Optional[List[Vehicle]]

RECORDING_PATH = "./recordings/"
LOG_FILE = os.path.join(RECORDING_PATH, "unified_events_log.json")

# Configure Logging
os.makedirs(RECORDING_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RECORDING_PATH, "offline_analyzer.log")),
        logging.StreamHandler()
    ]
)

def get_middle_frame(video_path):
    """Extracts the exact middle frame of the video where the subject is most likely centered."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        temp_img_path = os.path.join(RECORDING_PATH, "temp_ai_frame.jpg")
        cv2.imwrite(temp_img_path, frame)
        return temp_img_path
    return None

import requests
import base64
import json

def analyze_frame_with_gemini(image_path):
    """Uses Local Ollama (llama3.2-vision). Kept the function name `analyze_frame_with_gemini` to avoid breaking downstream loop logic."""
    logging.info(f"🧠 Asking Local Ollama (llama3.2-vision) to analyze the frame...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        prompt = (
            "You are an expert security surveillance AI. Look at this image closely. "
            "NOTE: This image may be from an Infrared Night-Vision camera (black and white). "
            "1. Identify the specific make and model of any vehicles. "
            "2. Identify the color of the vehicles (if night vision, use 'dark' or 'light'). "
            "3. Identify the clothing type and color of any people present (if night vision, describe the shade). "
            "CRITICAL: If there are NO vehicles, the 'vehicles' list MUST be empty. If there are NO people, the 'people' list MUST be empty. "
            "Output ONLY valid JSON exactly matching this structure with no markdown or extra text: "
            "{\"people\": [{\"clothing\": \"string\", \"complexion\": \"string\"}], \"vehicles\": [{\"make\": \"string\", \"model\": \"string\", \"color\": \"string\", \"type\": \"string\"}]}"
        )

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2-vision",
            "prompt": prompt,
            "images": [base64_image],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1}
        })
        
        response.raise_for_status()
        data = response.json()
        
        try:
            parsed_data = json.loads(data["response"])
        except json.JSONDecodeError:
            return {"error": "Ollama returned malformed JSON."}
        
        # Clean up empty arrays
        if 'people' in parsed_data and not parsed_data['people']:
            del parsed_data['people']
        if 'vehicles' in parsed_data and not parsed_data['vehicles']:
            del parsed_data['vehicles']
            
        logging.info(f"✅ Result: {parsed_data}")
        return parsed_data

    except Exception as e:
        error_msg = f"Ollama Error: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}

def run_analyzer():
    logging.info("="*50)
    logging.info("🚀 Tapo Offline AI Analyzer Started")
    logging.info("="*50)
    
    if not os.path.exists(LOG_FILE):
        logging.info("No unified events log found. Exiting.")
        return
        
    try:
        with open(LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read log file: {e}")
        return
        
    modified = False
    processed_count = 0
    
    # Look for events that haven't been analyzed yet
    for event in log_data:
        if "ai_analysis" not in event and "video_file" in event:
            video_path = os.path.join(RECORDING_PATH, event["video_file"])
            
            if not os.path.exists(video_path):
                logging.warning(f"⚠️ Video missing: {event['video_file']}. Tagging as error to prevent infinite retries.")
                event["ai_analysis"] = {"error": "Video file deleted or missing"}
                modified = True
                continue
            
            logging.info(f"🎬 Processing new event: {event['video_file']}")
            
            img_path = get_middle_frame(video_path)
            if img_path:
                explanation = analyze_frame_with_gemini(img_path)
                os.remove(img_path) # Clean up temp image
                
                event["ai_analysis"] = explanation
                modified = True
                processed_count += 1
                logging.info(f"✅ Result: {explanation}")

    # If we successfully analyzed new images, save them back to the JSON file
    if modified:
        try:
            # We RE-READ the latest json just in case the main camera app recorded 
            # a new event while Ollama was thinking (preventing accidental deletion of new records)
            with open(LOG_FILE, 'r') as f:
                latest_data = json.load(f)
            
            for latest_event in latest_data:
                for processed_event in log_data:
                    # Match records together safely
                    if (latest_event.get("timestamp") == processed_event.get("timestamp") and 
                        latest_event.get("cam") == processed_event.get("cam")):
                        if "ai_analysis" in processed_event:
                            latest_event["ai_analysis"] = processed_event["ai_analysis"]
                            
            with open(LOG_FILE, 'w') as f:
                json.dump(latest_data, f, indent=4)
            
            logging.info(f"💾 Successfully saved {processed_count} new AI analyses to JSON log.")
        except Exception as e:
            logging.error(f"⚠️ Failed to save JSON: {e}")
    else:
        logging.info("No new video events to analyze.")
        
    logging.info("🏁 Analysis Complete. Shutting down offline analyzer.")

if __name__ == "__main__":
    run_analyzer()
