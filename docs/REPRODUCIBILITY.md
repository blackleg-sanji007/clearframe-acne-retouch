# Setup, training, and reproducibility

[Back to the README](../README.md)

## 1. Install the environment

The supported, locally exercised configuration is Windows with Python 3.12, a 6 GB RTX 3050 Laptop GPU, and 16 GB system RAM. CUDA-enabled PyTorch 2.7.1+cu128 and Ultralytics 8.3.228 were used for the recorded run.

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup script downloads `uv` into `.tools`, installs a local Python 3.12 runtime into `.runtime`, creates `.venv`, installs the specified PyTorch/torchvision binaries and pinned direct dependencies, and runs `check_environment.py`. It does not require a system Python installation or a separate CUDA toolkit. The NVIDIA driver must support the selected PyTorch CUDA runtime. Setup needs several GB of download and disk capacity.

CPU alternative:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -CpuOnly
```

The script leaves an existing virtual environment in place. Use a separate checkout/environment to compare CPU and CUDA installations rather than assuming that rerunning setup will replace every installed package variant.

### Manual installation

With Python 3.12 already installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

Replace the PyTorch index with `https://download.pytorch.org/whl/cpu` for a fresh CPU environment. The direct requirements are pinned; transitive resolution, drivers, and CUDA hardware are not a guarantee of bitwise reproducibility.

## 2. Verify the included detector

`models/acne.pt` is the locally trained detector, not the original course project's checkpoint. Its portable metadata and SHA-256 checksum are in [models/acne.json](../models/acne.json). Training-time path and Git-origin metadata were sanitized for publication; all model tensors were checked for equality before and after export.

```powershell
.\.venv\Scripts\python.exe -c "import hashlib,json; from pathlib import Path; m=json.loads(Path('models/acne.json').read_text()); actual=hashlib.sha256(Path('models/acne.pt').read_bytes()).hexdigest(); assert actual==m['sha256']; print('Checkpoint checksum verified.')"
```

The file is approximately 22.5 MB and is included directly in Git. No Git LFS setup is required. It is a PyTorch/Ultralytics checkpoint; load checkpoints only from sources you trust.

## 3. Download optional DeepFill assets

```powershell
.\.venv\Scripts\python.exe download_assets.py --deepfill
```

This fetches the separately maintained port into `vendor/deepfillv2` at commit `74c4e0c722ee0c017c29bf2ab422766ee5b8b746` and places its generator weights at `models/deepfillv2_WGAN.pth`. The download code verifies the expected SHA-256:

```text
60d338fa0c7556db043052f1d05d964c9b817687990676096af32dbfc1c48e22
```

Both the third-party source checkout and these pretrained weights are ignored by Git. They are not bundled with Clearframe or covered by its project-source license. The source and download locations are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Google Drive can impose access or quota limits. Retry later if that occurs; the app can still use Fast mode without DeepFill. A checksum failure raises an error rather than silently accepting a different file. Remove the reported bad/partial file before retrying that download.

## 4. Launch and check the app

```powershell
.\.venv\Scripts\python.exe demo.py
# Or use the convenience script:
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

Open http://127.0.0.1:7860. To use another port or prevent browser auto-opening:

```powershell
.\.venv\Scripts\python.exe demo.py --port 7861 --no-browser
```

The server binds to localhost and does not create a public Gradio link. CUDA startup prepares the detector and, when present, DeepFill. Subsequent uploads reuse these objects. CPU mode defaults to Fast. `--no-warmup` skips preparation but makes the first applicable request slower.

Unit tests need neither dataset downloads nor DeepFill weights:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For the optional end-to-end test, run an app with DeepFill installed, then in another terminal:

```powershell
.\.venv\Scripts\python.exe tests/smoke_app.py --url http://127.0.0.1:7860 --image "C:\path\to\face-with-spots.jpg"
```

Choose an image with detectable spots; the smoke check intentionally fails if there are none. If ACNE04 is prepared, omitting `--image` selects a sample from its test split. The check tests both restoration modes, exact outside-mask preservation, and empty-selection behavior. It is a functional check, not an image-realism benchmark.

## 5. Obtain and prepare ACNE04

The [dataset authors](https://github.com/xpwu95/LDL) provide a [Google Drive folder](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ). The authors specify academic usage; photographs are downloaded separately and are not redistributed here.

```powershell
.\.venv\Scripts\python.exe download_assets.py --dataset
.\.venv\Scripts\python.exe prepare_dataset.py
```

The classification archive provides photographs (approximately 1.1 GB); the detection archive provides XML boxes (approximately 6 MB). Archives are retained in `data/downloads`, raw extraction uses `data/raw/acne04`, and prepared data uses `data/acne04`.

Preparation checks dimensions and box validity, removes exact-image duplicates, and creates seed-42 severity-stratified train/validation/test splits. The expected counts are 993 / 283 / 143. Compare the generated manifest with [reports/split_manifest.json](../reports/split_manifest.json), which records the published experiment's images and hashes.

An existing output directory is never overwritten. Reuse its `data.yaml`, or create a new output path and pass that path explicitly to training:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --output data/acne04-alternative --seed 42
.\.venv\Scripts\python.exe train.py --data data/acne04-alternative/data.yaml --no-publish
```

The generated `data.yaml` contains the local absolute dataset path. Regenerate or update it when moving prepared data to another checkout. Published manifests contain no machine-specific source path.

## 6. Train a new detector

Close the demo first. The normal training command starts from COCO-pretrained YOLOv8s:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 60 --batch 8 --workers 0 --imgsz 640 --name acne_local
```

The first run downloads the COCO checkpoint if missing. Model weights, optimizer state, metrics, and periodic checkpoints are saved under `runs/detect/<run-name>`. Ultralytics increments the name if the directory exists; use the actual path printed by training.

When successful, `train.py` copies the best validation checkpoint into `models/acne.pt` and writes new metadata. Restart the app to load it. Use `--no-publish` for exploratory runs that should leave the included app checkpoint alone.

Early stopping with patience 15 can finish before epoch 60. The published run stopped at 53, with its best validation fitness at 38. The supplied checkpoint was selected on validation mAP@0.50:0.95, not test performance.

### Resume an interruption

```powershell
.\.venv\Scripts\python.exe train.py --resume runs/detect/acne_local/weights/last.pt --epochs 60
```

Resume restores optimizer state and starts at the next saved epoch; `--epochs 60` means 60 total epochs. Omitting `--hours` removes any old optional time cap. Batch size, image size, device, worker count, and patience follow the command's supported resume overrides. An interrupted epoch is repeated from the last completed checkpoint.

For an optional supervisor that resumes an interrupted run, evaluates it, restarts the app, and runs its API check:

```powershell
.\.venv\Scripts\python.exe training_job.py --resume runs/detect/acne_local/weights/last.pt --epochs 60
```

This command requires an existing resumable checkpoint and an installed DeepFill model for its final two-mode API check. It records progress in `outputs/training-status.json` and logs under `.cache`. On Windows it prevents idle sleep during the job, but cannot prevent shutdowns, restarts, or lid-close actions. Keep the command's terminal open, or launch it using a hidden `Start-Process` if you need a background job; a restart still requires a manual resume.

Completed Ultralytics checkpoints have optimizer state stripped and are not resumable training sessions. To continue learning from one, start a new fine-tuning run:

```powershell
.\.venv\Scripts\python.exe train.py --model models/acne.pt --name acne_finetune --no-publish
```

If training already finished but final evaluation/app checks need to be run separately, use `training_job.py --finalize-only`. It evaluates the currently published detector and does not train again.

## 7. Evaluate

```powershell
.\.venv\Scripts\python.exe evaluate.py --weights models/acne.pt --output outputs/test-metrics.json
```

The evaluator uses the held-out `test` split, batch 2, workers 0, and a 640 px image-size target. It reports detection precision, recall, mAP@0.50, and mAP@0.50:0.95. It also writes curves and plots under `runs/evaluate`. None of these scores measures restoration realism.

The published metrics and curve CSV are preserved under `reports` so a new training/evaluation job does not overwrite the historical evidence. They describe the included checkpoint, not an arbitrary future locally replaced model.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Port 7860 is already in use | Open the already running app, stop that process, or use `--port 7861` |
| CUDA unavailable | Run `check_environment.py`; check the NVIDIA driver and that PyTorch is the CUDA build, or use `--device cpu` |
| GPU memory full during training | Close the demo and other GPU workloads; retry batch 4 or 2, or consider image size 512 / `yolov8n.pt` |
| DeepFill missing | Run `download_assets.py --deepfill`, or choose Fast mode |
| Model sees no spots | Use a clear, well-lit photo and lower confidence cautiously; review added detections |
| Retouch changes an unwanted feature | Deselect that box, reduce padding, and run restoration again |
| Training process stopped after a restart | Resume from the actual run's `weights/last.pt`; the status file may be stale |

Training-time and inference-time memory and speed depend on the laptop's power state, drivers, and background workload. A fixed seed, pinned direct dependencies, and recorded splits improve repeatability but do not guarantee identical results after interruptions or on another machine.
