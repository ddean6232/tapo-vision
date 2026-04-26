import os
import json
import glob
from datetime import datetime
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)
RECORDINGS_DIR = "/app/recordings"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tapo Vision Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #ffffff; padding: 20px; margin: 0; }
        h1 { color: #00e676; text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; max-width: 1400px; margin: 0 auto; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .card h3 { margin: 0 0 10px 0; color: #ffffff; }
        video { width: 100%; border-radius: 8px; margin-top: 15px; background: #000; outline: none; }
        .meta { font-size: 0.85em; color: #888888; margin-top: 12px; display: flex; justify-content: space-between; }
        .tag { display: inline-block; background: #00e676; color: #000; padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: bold; margin-right: 5px; text-transform: capitalize; }
        .empty { text-align: center; color: #888; font-size: 1.2em; grid-column: 1 / -1; margin-top: 50px; }
    </style>
</head>
<body>
    <h1>📸 Tapo Vision Analytics</h1>
    <div class="grid">
        {% for ev in events %}
        <div class="card">
            <h3>{{ ev.cam }}</h3>
            <div>
                {% for obj in ev.objs %}
                <span class="tag">{{ obj }}</span>
                {% endfor %}
            </div>
            <video controls preload="metadata">
                <source src="/video/{{ ev.video_file }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="meta">
                <span>Confidence: {{ ev.conf }}%</span>
                <span>{{ ev.time }}</span>
            </div>
        </div>
        {% else %}
        <div class="empty">No detections found yet. Waiting for AI events...</div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    json_files = glob.glob(os.path.join(RECORDINGS_DIR, "*.json"))
    # Sort files by modification time, newest first
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    events = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            # Derive video filename
            base_name = os.path.basename(jf).replace(".json", "")
            video_file = f"{base_name}.mp4"
            
            # Formatting time
            mtime = os.path.getmtime(jf)
            dt_str = datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M:%S")
            
            # Only add if the video file actually exists
            if os.path.exists(os.path.join(RECORDINGS_DIR, video_file)):
                events.append({
                    "cam": data.get("cam", "Unknown Camera"),
                    "objs": data.get("objs", []),
                    "conf": int(data.get("conf", 0) * 100),
                    "time": dt_str,
                    "video_file": video_file
                })
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    return render_template_string(HTML_TEMPLATE, events=events)

@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(RECORDINGS_DIR, filename)

if __name__ == "__main__":
    print("Starting Tapo Vision Dashboard on port 38180...")
    app.run(host="0.0.0.0", port=38180)