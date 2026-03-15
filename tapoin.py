import os
from dotenv import load_dotenv
from pytapo import Tapo

load_dotenv()

USER_ADMIN = os.getenv("USER_ADMIN", "admin")
PASS_ADMIN = os.getenv("PASS_ADMIN")

# Attempt 1: The 'admin' + Cloud Password method
tapo = Tapo("192.168.8.102", USER_ADMIN, PASS_ADMIN)

try:
    print("Checking connection...")
    info = tapo.getBasicInfo()
    print("✓ SUCCESS! You are in.")
    print(f"Device: {info['device_info']['basic_info']['device_model']}")
except Exception as e:
    print(f"FAILED: {e}")
    print("\nNext Step: Ensure 'Tapo Lab > Third-Party Compatibility' is ON in the app.")