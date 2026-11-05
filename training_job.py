"""Durable local job: resume training, evaluate, then restart and check the demo."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "outputs/training-status.json"
CACHE = ROOT / ".cache"
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_status(status):
    status["updated_utc"] = timestamp()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    # Windows readers can briefly deny replacement of an open status file.
    for attempt in range(50):
        try:
            temporary.replace(STATUS)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def read_metrics(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        try:
            epoch = int(float(row["epoch"]))
            score = float(row["metrics/mAP50(B)"])
            if math.isfinite(score):
                result.append((epoch, score))
        except (ValueError, TypeError, KeyError):
            continue
    return result


def stop_demo():
    matches = []
    for process in psutil.process_iter(["name"]):
        try:
            if (
                process.info["name"] and "python" in process.info["name"].lower()
                and "demo.py" in process.cmdline()
                and Path(process.exe()).is_relative_to(ROOT)
            ):
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for process in reversed(matches):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(matches, timeout=10)


def run_stage(command, log_path, status, phase, metrics_path=None):
    status.update(phase=phase, state="running", log=str(log_path))
    save_status(status)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=FLAGS,
        )
        status["active_process_pid"] = process.pid
        save_status(status)
        while process.poll() is None:
            if metrics_path:
                rows = read_metrics(metrics_path)
                if rows:
                    status["epochs_completed"] = rows[-1][0]
                    status["latest_validation_map50"] = rows[-1][1]
                    status["best_validation_map50"] = max(score for _, score in rows)
            status["elapsed_seconds"] = round(time.time() - status["started_timestamp"])
            save_status(status)
            time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"{phase} failed with exit code {process.returncode}; see {log_path}")


def start_demo(status):
    output_path = CACHE / "demo-trained.out.log"
    error_path = CACHE / "demo-trained.err.log"
    with output_path.open("w", encoding="utf-8") as output, error_path.open("w", encoding="utf-8") as error:
        process = subprocess.Popen(
            [sys.executable, "-u", "demo.py", "--no-browser"], cwd=ROOT,
            stdout=output, stderr=error, creationflags=FLAGS,
        )
    status["demo_pid"] = process.pid
    status["demo_url"] = "http://127.0.0.1:7860"
    status["phase"] = "starting_demo"
    save_status(status)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Demo startup failed; see {error_path}")
        try:
            with urllib.request.urlopen(status["demo_url"] + "/config", timeout=2) as response:
                json.load(response)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError(f"Demo did not become ready; see {output_path} and {error_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=ROOT / "runs/detect/acne_local/weights/last.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finalize-only", action="store_true", help="Evaluate and start the demo after training already finished")
    args = parser.parse_args()
    if not args.finalize_only and not args.resume.is_file():
        parser.error(f"Checkpoint missing: {args.resume}")
    if args.finalize_only and not (ROOT / "models/acne.pt").is_file():
        parser.error("Published weights are required for --finalize-only")
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        try:
            previous_process = psutil.Process(previous["job_pid"])
            if "training_job.py" in previous_process.cmdline():
                raise SystemExit(f"Training job already running as PID {previous_process.pid}")
        except (KeyError, psutil.NoSuchProcess):
            pass
    CACHE.mkdir(exist_ok=True)
    status = {
        "state": "running", "phase": "preparing", "job_pid": os.getpid(),
        "started_utc": timestamp(), "started_timestamp": time.time(),
        "target_epochs": args.epochs, "resume_checkpoint": str(args.resume.resolve()),
        "early_stopping_patience": args.patience, "time_limit_hours": None,
    }
    save_status(status)
    stop_demo()
    # Keep this training session awake while allowing the screen to turn off.
    # This cannot prevent a restart, shutdown, or closing the laptop lid.
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    for suffix in (".pt", ".json"):
        source = ROOT / f"models/acne{suffix}"
        if source.is_file() and not args.finalize_only:
            shutil.copy2(source, ROOT / f"models/acne_before_training{suffix}")
    try:
        if not args.finalize_only:
            run_stage(
                [sys.executable, "-u", "train.py", "--resume", str(args.resume.resolve()),
                 "--epochs", str(args.epochs), "--batch", str(args.batch),
                 "--workers", "0", "--patience", str(args.patience)],
                CACHE / "training-full.log", status, "training",
                args.resume.resolve().parents[1] / "results.csv",
            )
        rows = read_metrics(args.resume.resolve().parents[1] / "results.csv")
        if rows:
            status.update(epochs_completed=rows[-1][0], latest_validation_map50=rows[-1][1],
                          best_validation_map50=max(score for _, score in rows))
        report_path = ROOT / "outputs/trained-test-metrics.json"
        run_stage(
            [sys.executable, "-u", "evaluate.py", "--output", str(report_path)],
            CACHE / "evaluation-full.log", status, "evaluating",
        )
        status["test_metrics"] = json.loads(report_path.read_text(encoding="utf-8"))
        metadata_path = ROOT / "models/acne.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["test_metrics"] = status["test_metrics"]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        status["epochs_completed"] = metadata["epochs_completed"]
        status["weights"] = str(ROOT / "models/acne.pt")
        start_demo(status)
        run_stage(
            [sys.executable, "-u", "tests/smoke_app.py"],
            CACHE / "api-trained.log", status, "checking_demo",
        )
        status.update(state="complete", phase="complete", finished_utc=timestamp())
        status.pop("active_process_pid", None)
        save_status(status)
        print("Training, evaluation and demo checks completed.", flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error), failed_utc=timestamp())
        save_status(status)
        traceback.print_exc()
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
