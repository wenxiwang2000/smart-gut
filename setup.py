import subprocess, sys, pathlib

print("=" * 50)
print(" Smart Gut - First Time Setup ")
print("=" * 50)

root = pathlib.Path(__file__).parent.resolve()
req_file = root / "requirements.txt"

# Upgrade pip (optional but recommended)
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
except Exception as e:
    print(f"Warning: could not upgrade pip: {e}")

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])

print("\nSetup complete! You can now run:")
print(f"  python {root / 'main.py'}")



