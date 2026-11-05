"""Reusable detector and restoration models, independent of the web UI."""
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO, settings

from acne_demo.inpainting import DeepFill, blend_inside_mask, fast_inpaint

ROOT = Path(__file__).resolve().parents[1]
settings.update({"sync": False})


def default_weights():
    return ROOT / "models" / "acne.pt"


def normalize_image(image, max_side=1600):
    if image is None:
        raise ValueError("Upload a photo first.")
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Use an RGB or RGBA image with 8-bit pixels.")
        image = Image.fromarray(array).convert("RGB")
    original_size = image.size
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image).copy(), original_size


def make_mask(shape, boxes, selected=None, padding=2):
    if not 0 <= int(padding) <= 20:
        raise ValueError("Padding must be between 0 and 20 pixels.")
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    indices = range(len(boxes)) if selected is None else selected
    for index in indices:
        if not 0 <= index < len(boxes):
            raise ValueError("The selected spots are out of date. Detect spots again.")
        x1, y1, x2, y2 = boxes[index][:4]
        x1, y1 = max(0, int(np.floor(x1))-padding), max(0, int(np.floor(y1))-padding)
        x2, y2 = min(w, int(np.ceil(x2))+padding), min(h, int(np.ceil(y2))+padding)
        if x2 > x1 and y2 > y1:
            # Rectangles ensure coverage of the complete detected lesion.
            mask[y1:y2, x1:x2] = 255
    return mask


@dataclass
class Detection:
    rgb: np.ndarray
    boxes: list
    original_size: tuple
    elapsed: float


class AcnePipeline:
    def __init__(self, weights=None, device="auto"):
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.weights = Path(weights or default_weights()).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Detector weights missing: {self.weights}")
        self.lock = RLock()
        self.detector = YOLO(str(self.weights))
        if self.detector.task != "detect":
            raise ValueError("Select an acne object detection checkpoint.")
        names = [str(name).lower() for name in self.detector.names.values()]
        if len(names) != 1 or not any(word in names[0] for word in ("pimple", "acne", "lesion")):
            raise ValueError("Expected a single-class acne detector, not a generic pretrained YOLO model.")
        self.deepfill = None

    def warmup(self, include_deepfill=True):
        """Pay model setup and kernel initialization costs before accepting uploads."""
        image = np.full((512, 512, 3), 127, dtype=np.uint8)
        self.detect(image, confidence=0.95)
        if include_deepfill:
            with self.lock:
                if self.deepfill is None:
                    self.deepfill = DeepFill(self.device)
                mask = np.zeros((512, 512), dtype=np.uint8)
                mask[240:272, 240:272] = 255
                self.deepfill(image, mask)

    def detect(self, image, confidence=0.25):
        if not 0.01 <= float(confidence) <= 0.95:
            raise ValueError("Detection confidence must be between 0.01 and 0.95.")
        rgb, original_size = normalize_image(image)
        start = time.perf_counter()
        with self.lock, torch.inference_mode():
            result = self.detector.predict(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                conf=float(confidence), iou=0.4, imgsz=640,
                device=self.device, max_det=500, verbose=False,
            )[0]
            boxes = result.boxes.data.cpu().numpy().tolist()
        # Stable numbering from top to bottom helps with visual review.
        boxes.sort(key=lambda b: (b[1], b[0]))
        return Detection(rgb, boxes, original_size, time.perf_counter()-start)

    def restore(self, detection, selected=None, padding=2, strength=1.0, mode="DeepFillv2"):
        mask = make_mask(detection.rgb.shape, detection.boxes, selected, int(padding))
        start = time.perf_counter()
        if not np.any(mask) or float(strength) == 0:
            return detection.rgb.copy(), mask, 0.0
        with self.lock:
            if mode == "DeepFillv2":
                if self.deepfill is None:
                    self.deepfill = DeepFill(self.device)
                filled = self.deepfill(detection.rgb, mask)
            elif mode == "Fast (OpenCV)":
                filled = fast_inpaint(detection.rgb, mask)
            else:
                raise ValueError("Choose DeepFillv2 or Fast (OpenCV).")
        output = blend_inside_mask(detection.rgb, filled, mask, float(strength))
        return output, mask, time.perf_counter()-start


def draw_detections(detection, selected=None):
    output = detection.rgb.copy()
    indices = range(len(detection.boxes)) if selected is None else selected
    for i in indices:
        x1, y1, x2, y2, score, _ = detection.boxes[i]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(output, (x1, y1), (x2, y2), (34, 197, 154), 2)
        cv2.putText(output, str(i+1), (x1, max(16, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 154), 2)
    return output
