#!/bin/bash

# Path for the macOS launch agent
PLIST_PATH="$HOME/Library/LaunchAgents/com.darrendean.tapovision.plist"

# Generate the .plist file
cat << EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.darrendean.tapovision</string>
    <key>ProgramArguments</key>
    <array>
        <!-- Use the absolute path to the virtual environment's python directly to avoid PATH issues -->
        <string>/Users/darren_dean/Desktop/tapo-vision/.venv/bin/python</string>
        <string>/Users/darren_dean/Desktop/tapo-vision/my_tapo_ai.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/darren_dean/Desktop/tapo-vision</string>
    
    <!-- Start automatically when Darren logs in -->
    <key>RunAtLoad</key>
    <true/>
    
    <!-- Automatically restart it if it ever crashes -->
    <key>KeepAlive</key>
    <true/>
    
    <!-- Logging -->
    <key>StandardOutPath</key>
    <string>/Users/darren_dean/Desktop/tapo-vision/tapo_service.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/darren_dean/Desktop/tapo-vision/tapo_service.err</string>
</dict>
</plist>
EOF

# Load the service into Apple's launchd system
launchctl load "$PLIST_PATH"

echo "✅ Tapo-Vision has been successfully registered as a macOS service!"
echo "It will automatically start in the background (and the Menu Bar) whenever you log in."
echo ""
echo "To temporarily stop the service permanently, you can use the Menu Bar 'Quit' button."
echo "To completely unregister the auto-start, run:"
echo "launchctl unload $PLIST_PATH"
