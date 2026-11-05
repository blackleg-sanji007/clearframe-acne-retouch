"""Local acne detection and photo retouching demo."""
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_config = _root / ".cache" / "ultralytics"
_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_config))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
