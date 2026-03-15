import os
from dotenv import load_dotenv
from pytapo import Tapo
import time

# --- CONFIGURATION ---
load_dotenv()
IP_ADDRESS = "192.168.8.102" 
USER = os.getenv("USER_ADMIN", "admin")
PASS = os.getenv("PASS_ADMIN")

print(f"Connecting to {IP_ADDRESS} as {USER}...")

try:
    tapo = Tapo(IP_ADDRESS, USER, PASS)
    
    # Verify we can talk to it
    print(f"✓ Connected to {tapo.getBasicInfo()['device_info']['basic_info']['device_model']}")

    # Let's do a simple dance: Left, Right, Up, Down
    test_moves = [
        ("Left", -40, 0),
        ("Right", 80, 0),
        ("Back to Center", -40, 0),
        ("Up", 0, 20),
        ("Down", 0, -40),
        ("Back to Horizon", 0, 20)
    ]

    for label, x, y in test_moves:
        print(f"Action: {label}...")
        tapo.moveMotor(x, y)
        time.sleep(1.5) # The motor needs time to physically travel

    print("\n✓ Test Complete! The camera is responsive.")

except Exception as e:
    print(f"ERROR: {e}")