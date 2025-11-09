from pathlib import Path
from ultralytics import YOLO

# Path to your YOLO model
p = Path("models/best_scratch.pt")

print("exists:", p.exists())

if not p.exists():
    raise FileNotFoundError(f"Model not found at {p}")

# Load model
m = YOLO(str(p))
print("✅ Model loaded successfully!")
print("Labels:", getattr(m, "names", None))
