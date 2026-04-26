import os
import json
import glob
from datetime import datetime
from flask import Flask, render_template_string, send_from_directory, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
RECORDINGS_DIR = os.getenv("RECORDING_PATH", "./recordings/")

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
        .card { background: #1e1e1e; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); position: relative; transition: transform 0.2s, opacity 0.3s; }
        .card:hover { transform: translateY(-2px); }
        .card h3 { margin: 0 0 10px 0; color: #ffffff; }
        video { width: 100%; border-radius: 8px; margin-top: 15px; background: #000; outline: none; }
        .meta { font-size: 0.85em; color: #888888; margin-top: 12px; display: flex; justify-content: space-between; align-items: center; }
        .tag { display: inline-block; background: #00e676; color: #000; padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: bold; margin-right: 5px; text-transform: capitalize; }
        .empty { text-align: center; color: #888; font-size: 1.2em; grid-column: 1 / -1; margin-top: 50px; display: none; }
        .delete-btn { position: absolute; top: 12px; right: 12px; background: #ff4444; color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 0.8em; font-weight: bold; transition: background 0.2s; z-index: 10; }
        .delete-btn:hover { background: #cc0000; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 100; justify-content: center; align-items: center; }
        .modal-content { background: #1e1e1e; padding: 25px; border-radius: 12px; max-width: 500px; width: 90%; color: #fff; line-height: 1.5; position: relative; }
        .modal-close { position: absolute; top: 15px; right: 15px; background: transparent; color: #ff4444; border: none; font-size: 1.5em; cursor: pointer; font-weight: bold; }
        .ai-desc { font-size: 0.9em; color: #aaa; margin-top: 15px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { color: #00e676; cursor: pointer; font-size: 0.85em; font-weight: bold; margin-top: 5px; display: inline-block; }
    </style>
</head>
<body>
    <h1>📸 Tapo Vision Analytics</h1>
    <div id="video-grid" class="grid"></div>
    <div id="empty-msg" class="empty">No detections found yet. Waiting for AI events...</div>

    <!-- Modal for full AI Analysis -->
    <div id="ai-modal" class="modal-overlay" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal(event)">×</button>
            <h2 style="margin-top:0; color:#00e676;">Full AI Analysis</h2>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        let currentEvents = []; // Global store for modal data

        // Fetch events silently in the background
        async function pollEvents() {
            try {
                const res = await fetch('/api/events');
                const events = await res.json();
                
                const grid = document.getElementById('video-grid');
                const emptyMsg = document.getElementById('empty-msg');
                
                // Keep track of what is currently on the screen
                const existingCards = Array.from(grid.children).map(c => c.dataset.basename);
                const fetchedCards = events.map(e => e.base_name);

                currentEvents = events; // Update global store

                // Add NEW recordings seamlessly
                // Reverse the array locally so we prepend the newest ones at the top correctly
                [...events].reverse().forEach(ev => {
                    if (!existingCards.includes(ev.base_name)) {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.dataset.basename = ev.base_name;
                        
                        let tagsHtml = ev.objs.map(obj => `<span class="tag">${obj}</span>`).join('');
                        let aiHtml = '';
                        if (ev.ai_desc) {
                            aiHtml = `
                                <div class="ai-desc">${ev.ai_desc}</div>
                                <div class="read-more" onclick="openModal('${ev.base_name}')">Read full analysis...</div>
                            `;
                        }
                        
                        card.innerHTML = `
                            <button class="delete-btn" onclick="deleteRecording('${ev.base_name}')">Delete</button>
                            <h3>${ev.cam}</h3>
                            <div>${tagsHtml}</div>
                            <video controls preload="metadata">
                                <source src="/video/${ev.video_file}" type="video/mp4">
                            </video>
                            ${aiHtml}
                            <div class="meta">
                                <span>Confidence: ${ev.conf}%</span>
                                <span>${ev.time}</span>
                            </div>
                        `;
                        // Insert at the very top of the grid
                        grid.insertBefore(card, grid.firstChild);
                    }
                });

                // Remove deleted cards (if they were deleted from the hard drive externally)
                existingCards.forEach(baseName => {
                    if (!fetchedCards.includes(baseName)) {
                        const cardToRemove = document.querySelector(`.card[data-basename="${baseName}"]`);
                        if(cardToRemove) cardToRemove.remove();
                    }
                });

                // Toggle empty message
                if (events.length === 0) {
                    emptyMsg.style.display = 'block';
                } else {
                    emptyMsg.style.display = 'none';
                }
            } catch (err) {
                console.error("Error fetching events:", err);
            }
        }

        // Delete Function (Instant UI removal)
        function deleteRecording(baseName) {
            if(confirm("Are you sure you want to delete this recording?")) {
                fetch('/delete/' + baseName, { method: 'POST' })
                .then(response => {
                    if(response.ok) {
                        // Instantly make it vanish from the screen
                        const card = document.querySelector(`.card[data-basename="${baseName}"]`);
                        if(card) {
                            card.style.opacity = '0';
                            setTimeout(() => card.remove(), 300); // Wait for fade out
                        }
                    } else {
                        alert("Failed to delete the recording.");
                    }
                });
            }
        }

        // Modal Functions
        function openModal(baseName) {
            const ev = currentEvents.find(e => e.base_name === baseName);
            if (ev && ev.ai_desc_full) {
                document.getElementById('modal-body').innerHTML = ev.ai_desc_full;
                document.getElementById('ai-modal').style.display = 'flex';
            }
        }

        function closeModal(e) {
            document.getElementById('ai-modal').style.display = 'none';
        }

        // Poll every 5 seconds invisibly
        setInterval(pollEvents, 5000);
        // Load immediately on page load
        pollEvents();
    </script>
</body>
</html>
"""

def get_events_list():
    json_files = glob.glob(os.path.join(RECORDINGS_DIR, "*.json"))
    # Sort files by modification time, newest first
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    events = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            base_name = os.path.basename(jf).replace(".json", "")
            video_file = f"{base_name}.mp4"
            
            mtime = os.path.getmtime(jf)
            dt_str = datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M:%S")
            
            # Format AI Description if it exists
            ai_desc = ""
            ai_desc_full = ""
            if "ai_analysis" in data:
                analysis = data["ai_analysis"]
                parts = []
                
                # Full HTML Description for Modal
                if "people" in analysis:
                    ai_desc_full += "<h4>👥 People Detected:</h4><ul>"
                    for p in analysis["people"]:
                        clothing = p.get('clothing', 'unknown')
                        complexion = p.get('complexion', 'unknown')
                        parts.append(f"Person ({clothing})")
                        ai_desc_full += f"<li>Clothing: {clothing.title()}<br>Complexion: {complexion.title()}</li>"
                    ai_desc_full += "</ul>"
                    
                if "vehicles" in analysis:
                    ai_desc_full += "<h4>🚗 Vehicles Detected:</h4><ul>"
                    for v in analysis["vehicles"]:
                        v_str = f"{v.get('color', '')} {v.get('make', '')} {v.get('model', '')}".strip()
                        parts.append(f"Vehicle ({v_str})")
                        ai_desc_full += f"<li>{v_str.title()} (Type: {v.get('type', 'unknown').title()})</li>"
                    ai_desc_full += "</ul>"
                    
                if parts:
                    ai_desc = " • ".join(parts)
            
            if os.path.exists(os.path.join(RECORDINGS_DIR, video_file)):
                events.append({
                    "cam": data.get("cam", "Unknown Camera"),
                    "objs": data.get("objs", []),
                    "conf": int(data.get("conf", 0) * 100),
                    "time": dt_str,
                    "video_file": video_file,
                    "base_name": base_name,
                    "ai_desc": ai_desc,
                    "ai_desc_full": ai_desc_full
                })
        except Exception as e:
            print(f"Error reading {jf}: {e}")
    return events

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/events")
def api_events():
    return jsonify(get_events_list())

@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(RECORDINGS_DIR, filename)

@app.route("/delete/<base_name>", methods=["POST"])
def delete_recording(base_name):
    base_name = os.path.basename(base_name)
    json_path = os.path.join(RECORDINGS_DIR, f"{base_name}.json")
    mp4_path = os.path.join(RECORDINGS_DIR, f"{base_name}.mp4")
    
    try:
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
        return "Deleted", 200
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    print("Starting Tapo Vision Dashboard on port 38180...")
    app.run(host="0.0.0.0", port=38180)