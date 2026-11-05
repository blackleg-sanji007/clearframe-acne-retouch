"""Fine-tune a YOLO detector locally and publish best weights for the demo."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("WANDB_MODE", "disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/acne04/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained weights; yolov8n.pt uses less memory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hours", type=float, default=None, help="Optional approximate training time budget in hours")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on this 16 GB RAM Windows laptop")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", default="acne_local")
    parser.add_argument("--device", default="0", help="0 for NVIDIA GPU; cpu for CPU")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from weights/last.pt")
    parser.add_argument("--no-publish", action="store_true", help="Do not replace app weights; use for smoke runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.workers < 0:
        parser.error("Epochs and batch must be positive; imgsz must be at least 32.")
    if args.hours is not None and args.hours <= 0:
        parser.error("Training hours must be positive.")
    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "wandb": False, "mlflow": False})
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run check_environment.py after setup, or pass --device cpu.")
    if not args.resume and not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. Run download_assets.py --dataset and prepare_dataset.py first.")
    start = time.perf_counter()
    model = YOLO(str(args.resume or args.model))
    try:
        if args.resume:
            from ultralytics.models.yolo.detect.train import DetectionTrainer

            class ResumeTrainer(DetectionTrainer):
                def check_resume(self, overrides):
                    super().check_resume(overrides)
                    if self.resume:
                        # Ultralytics otherwise restores the old timed-run cap
                        # and ignores epoch/time/worker overrides on resume.
                        self.args.epochs = args.epochs
                        self.args.time = args.hours
                        self.args.workers = args.workers
                        self.args.patience = args.patience

            def verify_resume(trainer):
                if trainer.start_epoch < 1 or not trainer.optimizer.state:
                    raise RuntimeError("The checkpoint did not restore the previous training/optimizer state.")
                print(
                    f"Resume verified: epoch {trainer.start_epoch+1}/{trainer.epochs}, "
                    f"time limit={trainer.args.time}, optimizer states={len(trainer.optimizer.state)}",
                    flush=True,
                )

            model.add_callback("on_train_start", verify_resume)
            model.train(
                trainer=ResumeTrainer, resume=True, device=args.device,
                workers=args.workers, batch=args.batch, imgsz=args.imgsz,
            )
        else:
            # General COCO weights avoid reusing a detector that may already have
            # seen the validation and test photos in the upstream project.
            model.train(
                data=str(args.data.resolve()), epochs=args.epochs,
                time=args.hours,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                amp=args.device != "cpu", workers=args.workers, cache=False,
                optimizer="AdamW", lr0=0.001, patience=args.patience,
                project=str(ROOT / "runs" / "detect"), name=args.name,
                seed=42, deterministic=True, save=True, save_period=5,
                degrees=8, translate=0.05, scale=0.25, fliplr=0.5,
                flipud=0, mosaic=0.5, close_mosaic=10, mixup=0,
                hsv_h=0.01, hsv_s=0.3, hsv_v=0.25, plots=True,
            )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("GPU memory full. Close the demo, then retry --batch 1. If needed use --imgsz 512 or --model yolov8n.pt.")
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nTraining finished in {(time.perf_counter()-start)/60:.1f} minutes.")
    print(f"Best checkpoint: {best}")
    if not best.is_file():
        raise SystemExit("No best checkpoint produced; app weights were not changed.")
    if not args.no_publish:
        target = ROOT / "models" / "acne.pt"
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix(".tmp")
        shutil.copy2(best, temporary)
        temporary.replace(target)
        metadata = {
            "checkpoint": str(best), "dataset": str(args.data.resolve()),
            "source_model": str(args.resume or args.model), "run": str(save_dir),
            "epochs_completed": model.trainer.epoch + 1,
            "target_epochs": args.epochs, "status": "training complete",
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"App weights saved to {target}. Restart the demo to load them.")
    print("Evaluate the untouched test set separately with evaluate.py.")


if __name__ == "__main__":
    main()
