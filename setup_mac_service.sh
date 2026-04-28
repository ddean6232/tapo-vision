#!/bin/bash

# Target directory in the user's home folder
TARGET_DIR="$HOME/tapo-vision"
PLIST_PATH="$HOME/Library/LaunchAgents/com.darrendean.tapovision.plist"

# Get the absolute path to uv so launchd knows exactly where to find it
UV_PATH=$(which uv)

if [ -z "$UV_PATH" ]; then
    echo "Error: Could not find 'uv' command. Make sure it is installed."
    exit 1
fi

cat << EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.darrendean.tapovision</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/caffeinate</string>
        <string>-is</string>
        <string>$UV_PATH</string>
        <string>run</string>
        <string>$TARGET_DIR/tapo_vision.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$TARGET_DIR</string>
    
    <!-- Start automatically when Darren logs in -->
    <key>RunAtLoad</key>
    <true/>
    
    <!-- Automatically restart it if it ever crashes -->
    <key>KeepAlive</key>
    <true/>
    
    <!-- Logging -->
    <key>StandardOutPath</key>
    <string>$TARGET_DIR/tapo_service.log</string>
    <key>StandardErrorPath</key>
    <string>$TARGET_DIR/tapo_service.err</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin</string>
    </dict>

    <!-- Increase open file limits to prevent OSError: [Errno 24] crashes -->
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>10240</integer>
    </dict>
    <key>HardResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>10240</integer>
    </dict>
</dict>
</plist>
EOF

# Load the service
launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl load "$PLIST_PATH"

echo "✅ Tapo-Vision has been successfully registered as a macOS background service!"
echo "It is now running invisibly in the background from $TARGET_DIR."
echo ""
echo "To view live logs, run: tail -f $TARGET_DIR/tapo_service.log"
echo "To stop the service permanently, run: launchctl unload $PLIST_PATH"