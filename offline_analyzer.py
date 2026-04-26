#!/usr/bin/env python3
"""
Tapo Offline AI Analyzer
Usage:
    python offline_analyzer.py                             # Default: ./recordings/
    python offline_analyzer.py --recordings /path/to/recs # Custom recordings path
    python offline_analyzer.py --model llama3.2-vision    # Different Ollama vision model
"""
import os
import sys
import json
import signal
import argparse
import logging
import base64
import requests
import cv2
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    logging.info(f"🛑 Received signal {signum}. Finishing current task then shutting down cleanly...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", default=os.getenv("RECORDING_PATH", "./recordings/"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2-vision"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def get_key_frames(video_path: str, recordings_path: str) -> str:
    """Extracts 5 frames from the middle and merges them into a single 'filmstrip' image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return None

    sample_points = [int(total_frames * pct) for pct in (0.30, 0.40, 0.50, 0.60, 0.70)]
    frames = []

    for frame_idx in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Resize down to save LLM context window/VRAM
            h, w = frame.shape[:2]
            new_w = 480
            new_h = int(new_w * h / w)
            resized = cv2.resize(frame, (new_w, new_h))
            frames.append(resized)

    cap.release()
    
    if not frames:
        return None
        
    # Stitch horizontally into one wide image
    collage = cv2.hconcat(frames)
    collage_path = os.path.join(recordings_path, f"temp_ai_collage.jpg")
    cv2.imwrite(collage_path, collage)
    return collage_path

def analyze_sequence(image_path: str, ollama_url: str, model: str) -> dict:
    """Sends the filmstrip to Ollama for superior contextual analysis."""
    logging.info(f"🧠 Asking {model} to analyze the 5-frame filmstrip...")
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "You are an expert security surveillance AI. Look at this single image carefully. "
            "It is a 'filmstrip' containing 5 sequential frames taken from a security video, reading left-to-right. "
            "By looking at how objects change across the 5 frames, you can perfectly identify moving vehicles or people. "
            "1. Identify the specific make and model of any vehicles. "
            "2. Identify the color of the vehicles (if night vision, use 'dark' or 'light'). "
            "3. Identify the clothing type and color of any people present. "
            "CRITICAL: If there are NO vehicles, the 'vehicles' list MUST be empty. "
            "If there are NO people, the 'people' list MUST be empty. "
            "Output ONLY valid JSON matching this structure exactly, no extra text or markdown: "
            "{\"people\": [{\"clothing\": \"string\", \"complexion\": \"string\"}], "
            "\"vehicles\": [{\"make\": \"string\", \"model\": \"string\", \"color\": \"string\", \"type\": \"string\"}]}"
        )

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [base64_image],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=180
        )
        response.raise_for_status()
        data = response.json()

        try:
            parsed = json.loads(data["response"])
        except (json.JSONDecodeError, KeyError):
            logging.error("⚠️ Ollama returned malformed JSON.")
            return {"error": "Ollama returned malformed JSON."}

        if "people" in parsed and not parsed["people"]:
            del parsed["people"]
        if "vehicles" in parsed and not parsed["vehicles"]:
            del parsed["vehicles"]

        return parsed

    except requests.exceptions.ConnectionError:
        msg = f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?"
        logging.error(msg)
        return {"error": msg}
    except Exception as e:
        logging.error(f"Ollama Error: {e}")
        return {"error": str(e)}

def run_analyzer(recordings_path: str, model: str, ollama_url: str):
    recordings_path = os.path.abspath(recordings_path)

    logging.info("=" * 50)
    logging.info("🚀 Tapo Offline AI Analyzer Started")
    logging.info(f"   📁 Recordings : {recordings_path}")
    logging.info(f"   🤖 Model      : {model}")
    logging.info(f"   🌐 Ollama URL : {ollama_url}")
    logging.info("=" * 50)

    import glob
    json_files = glob.glob(os.path.join(recordings_path, "*.json"))
    processed_count = 0

    for jf in json_files:
        if _shutdown_requested:
            logging.info("🛑 Shutdown requested — saving progress and exiting.")
            break

        try:
            with open(jf, "r") as f:
                event = json.load(f)
        except Exception as e:
            logging.error(f"Failed to read {jf}: {e}")
            continue
            
        # Ignore legacy files like unified_events_log.json which are lists, not dicts
        if not isinstance(event, dict):
            continue

        if "ai_analysis" not in event:
            base_name = os.path.basename(jf).replace(".json", "")
            video_path = os.path.join(recordings_path, f"{base_name}.mp4")

            if not os.path.exists(video_path):
                logging.warning(f"⚠️ Video missing: {video_path}. Tagging to prevent retries.")
                event["ai_analysis"] = {"error": "Video file deleted or missing"}
                with open(jf, "w") as f:
                    json.dump(event, f, indent=4)
                continue

            logging.info(f"🎬 Processing: {base_name}.mp4 (5-frame filmstrip)")
            img_path = get_key_frames(video_path, recordings_path)

            if img_path:
                aggregated = analyze_sequence(img_path, ollama_url, model)
                try:
                    os.remove(img_path)
                except OSError:
                    pass

                logging.info(f"✅ AI Result: {aggregated}")
                event["ai_analysis"] = aggregated
                with open(jf, "w") as f:
                    json.dump(event, f, indent=4)
                    
                processed_count += 1

    logging.info(f"🏁 Analysis complete. Analyzed {processed_count} new videos.")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.recordings, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.recordings, "offline_analyzer.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run_analyzer(args.recordings, args.model, args.ollama_url)