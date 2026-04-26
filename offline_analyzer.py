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
    """Extracts 3 strategically spaced frames (20%, 50%, 80%) for multi-pass analysis."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 3:
        cap.release()
        return []

    # 20% = object entering, 50% = fully centered, 80% = best angle before leaving
    sample_points = [int(total_frames * pct) for pct in (0.20, 0.50, 0.80)]
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


def aggregate_results(results: list) -> dict:
    """Merges detections from multiple frame analyses into one clean result.
    
    Strategy:
    - Vehicles: deduplicate by (make, type) key — keep the entry with most detail
    - People: keep all unique clothing descriptions (different people at diff times)
    """
    all_vehicles: dict = {}  # key = (make.lower, type.lower) -> best entry
    all_people:   list = []
    seen_clothing: set = set()

    for result in results:
        if not result or "error" in result:
            continue

        for v in result.get("vehicles", []):
            make  = (v.get("make")  or "unknown").strip()
            vtype = (v.get("type")  or "unknown").strip()
            key   = (make.lower(), vtype.lower())
            # Prefer entries that have real make/model over 'unknown'
            if key not in all_vehicles or make.lower() not in ("unknown", ""):
                all_vehicles[key] = v

        for p in result.get("people", []):
            clothing = (p.get("clothing") or "").strip().lower()
            if clothing and clothing not in seen_clothing:
                seen_clothing.add(clothing)
                all_people.append(p)

    merged: dict = {}
    if all_vehicles:
        merged["vehicles"] = list(all_vehicles.values())
    if all_people:
        merged["people"] = all_people
    return merged

# ── Vision analysis ─────────────────────────────────────────────────────────────
def analyze_frame(image_path: str, ollama_url: str, model: str) -> dict:
    """Sends the frame to Ollama and returns structured JSON analysis."""
    logging.info(f"🧠 Asking {model} to analyze the frame...")
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "You are an expert security surveillance AI. Look at this image closely. "
            "NOTE: This image may be from an Infrared Night-Vision camera (black and white). "
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
            timeout=120  # 2 minute timeout per frame
        )
        response.raise_for_status()
        data = response.json()

        try:
            parsed = json.loads(data["response"])
        except (json.JSONDecodeError, KeyError):
            logging.error("⚠️ Ollama returned malformed JSON.")
            return {"error": "Ollama returned malformed JSON."}

        # Strip empty arrays to keep the log clean
        if "people" in parsed and not parsed["people"]:
            del parsed["people"]
        if "vehicles" in parsed and not parsed["vehicles"]:
            del parsed["vehicles"]

        logging.info(f"✅ Result: {parsed}")
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
    log_file = os.path.join(recordings_path, "unified_events_log.json")

    logging.info("=" * 50)
    logging.info("🚀 Tapo Offline AI Analyzer Started")
    logging.info(f"   📁 Recordings : {recordings_path}")
    logging.info(f"   🤖 Model      : {model}")
    logging.info(f"   🌐 Ollama URL : {ollama_url}")
    logging.info("=" * 50)

    if not os.path.exists(log_file):
        logging.warning(f"No unified_events_log.json found at: {log_file}")
        return

    try:
        with open(log_file, "r") as f:
            log_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read log file: {e}")
        return

    modified = False
    processed_count = 0

    for event in log_data:
        if _shutdown_requested:
            logging.info("🛑 Shutdown requested — saving progress and exiting.")
            break

        if "ai_analysis" not in event and "video_file" in event:
            video_path = os.path.join(recordings_path, event["video_file"])

            if not os.path.exists(video_path):
                logging.warning(f"⚠️ Video missing: {event['video_file']}. Tagging to prevent infinite retries.")
                event["ai_analysis"] = {"error": "Video file deleted or missing"}
                modified = True
                continue

            logging.info(f"🎬 Processing: {event['video_file']} (3-frame multi-pass)")
            img_paths = get_key_frames(video_path, recordings_path)

            if img_paths:
                frame_results = []
                for i, img_path in enumerate(img_paths):
                    logging.info(f"  🖼️  Frame {i+1}/3...")
                    result = analyze_frame(img_path, ollama_url, model)
                    frame_results.append(result)
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

                aggregated = aggregate_results(frame_results)
                logging.info(f"✅ Aggregated: {aggregated}")
                event["ai_analysis"] = aggregated
                modified = True
                processed_count += 1

    # Safe-merge save: re-read the file to pick up any new events written by the
    # main camera app while we were processing, then overlay our AI results.
    if modified:
        try:
            with open(log_file, "r") as f:
                latest_data = json.load(f)

            result_map = {
                (e.get("timestamp"), e.get("cam")): e.get("ai_analysis")
                for e in log_data
                if "ai_analysis" in e
            }
            for event in latest_data:
                key = (event.get("timestamp"), event.get("cam"))
                if key in result_map:
                    event["ai_analysis"] = result_map[key]

            with open(log_file, "w") as f:
                json.dump(latest_data, f, indent=4)

            logging.info(f"💾 Saved {processed_count} new AI analyses to log.")
        except Exception as e:
            logging.error(f"⚠️ Failed to save JSON: {e}")
    else:
        logging.info("No new video events to analyze.")

    logging.info("🏁 Analysis complete. Shutting down.")

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
