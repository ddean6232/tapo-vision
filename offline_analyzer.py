#!/usr/bin/env python3
"""
Tapo Offline AI Analyzer
Usage:
    python offline_analyzer.py                             # Default: ./recordings/
    python offline_analyzer.py --recordings /path/to/recs # Custom recordings path
    python offline_analyzer.py --model llava:13b          # Different Ollama vision model
    python offline_analyzer.py --ollama-url http://192.168.1.10:11434  # Remote Ollama server
    python offline_analyzer.py --log-level DEBUG          # Verbose logging
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

# ── Graceful shutdown ───────────────────────────────────────────────────────────
# Handles Ctrl-C as well as kill/SIGTERM from launchd, Activity Monitor, etc.
_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    logging.info(f"🛑 Received signal {signum}. Finishing current task then shutting down cleanly...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── CLI Arguments ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tapo Offline AI Analyzer — locally analyzes security camera recordings using Ollama vision models."
    )
    parser.add_argument(
        "--recordings",
        default="./recordings/",
        help="Path to the recordings folder containing video files and unified_events_log.json. "
             "Default: ./recordings/"
    )
    parser.add_argument(
        "--model",
        default="llama3.2-vision",
        help="Ollama vision model to use for analysis. Default: llama3.2-vision"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama API server. Default: http://localhost:11434"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level. Default: INFO"
    )
    return parser.parse_args()

# ── Frame extraction ────────────────────────────────────────────────────────────
def get_key_frames(video_path: str, recordings_path: str) -> List[str]:
    """Extracts 5 strategically spaced frames from the middle (30% to 70%) for maximum object visibility."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return []

    # Focus on the middle of the event where the object is likely most visible
    sample_points = [int(total_frames * pct) for pct in (0.30, 0.40, 0.50, 0.60, 0.70)]
    saved_paths = []

    for i, frame_idx in enumerate(sample_points):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(recordings_path, f"temp_ai_frame_{i}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)

    cap.release()
    return saved_paths

# ── Vision analysis ─────────────────────────────────────────────────────────────
def analyze_sequence(image_paths: List[str], ollama_url: str, model: str) -> dict:
    """Sends all 5 frames to Ollama at once for superior contextual analysis."""
    logging.info(f"🧠 Asking {model} to analyze {len(image_paths)} sequential frames...")
    try:
        base64_images = []
        for path in image_paths:
            with open(path, "rb") as f:
                base64_images.append(base64.b64encode(f.read()).decode("utf-8"))

        prompt = (
            "You are an expert security surveillance AI. Look at these 5 sequential frames taken from the middle of a security video. "
            "By looking at the sequence, you can better identify moving objects. "
            "NOTE: This may be from an Infrared Night-Vision camera (black and white). "
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
                "images": base64_images,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=180  # Longer timeout since it's processing 5 images at once
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

# ── Main pipeline ───────────────────────────────────────────────────────────────
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

        if "ai_analysis" not in event:
            # Derive video filename
            base_name = os.path.basename(jf).replace(".json", "")
            video_path = os.path.join(recordings_path, f"{base_name}.mp4")

            if not os.path.exists(video_path):
                logging.warning(f"⚠️ Video missing: {video_path}. Tagging to prevent retries.")
                event["ai_analysis"] = {"error": "Video file deleted or missing"}
                with open(jf, "w") as f:
                    json.dump(event, f, indent=4)
                continue

            logging.info(f"🎬 Processing: {base_name}.mp4 (5-frame middle distribution)")
            img_paths = get_key_frames(video_path, recordings_path)

            if img_paths:
                aggregated = analyze_sequence(img_paths, ollama_url, model)
                for img_path in img_paths:
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

                logging.info(f"✅ AI Result: {aggregated}")
                
                # Save the new analysis back to the individual JSON file
                event["ai_analysis"] = aggregated
                with open(jf, "w") as f:
                    json.dump(event, f, indent=4)
                    
                processed_count += 1

    logging.info(f"🏁 Analysis complete. Analyzed {processed_count} new videos.")

# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    recordings_path = args.recordings
    os.makedirs(recordings_path, exist_ok=True)

    # Logging goes to both the recordings folder and stdout
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(recordings_path, "offline_analyzer.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run_analyzer(
        recordings_path=recordings_path,
        model=args.model,
        ollama_url=args.ollama_url,
    )
