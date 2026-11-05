"""Evaluate a local checkpoint on the held-out ACNE04 test split."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "models/acne.pt"))
    parser.add_argument("--data", default=str(ROOT / "data/acne04/data.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, help="Write test metrics to JSON")
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    result_directory = []
    model.add_callback("on_val_end", lambda validator: result_directory.append(str(validator.save_dir)))
    metrics = model.val(
        data=args.data, split="test", device=args.device,
        imgsz=640, batch=2, workers=0, plots=True,
        project=str(ROOT / "runs/evaluate"), name="test",
    )
    print(f"Test mAP50: {metrics.box.map50:.4f}; mAP50-95: {metrics.box.map:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "weights": str(Path(args.weights).resolve()), "split": "test",
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "results_directory": result_directory[-1],
        }
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
