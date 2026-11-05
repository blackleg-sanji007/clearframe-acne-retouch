# Recorded experiment

These files preserve evidence for the included `models/acne.pt` detector:

- `training.csv`: per-epoch Ultralytics metrics for the 53 completed epochs. The `time` column resets after resume and is not a continuous wall clock.
- `test_metrics.json`: held-out test metrics, selected epoch, and published model checksum.
- `split_manifest.json`: seed, split description, duplicate count, filenames, and SHA-256 hashes. Photographs and machine-specific paths are excluded.
- `plot_training.py`: regenerates `docs/assets/validation-curves.png` from the CSV using the selected epoch in `models/acne.json`.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe reports/plot_training.py
```

These are historical results. Replacing or retraining the app checkpoint does not automatically rewrite these published reports. See [EXPERIMENTS.md](../docs/EXPERIMENTS.md) for interpretation and limitations.
