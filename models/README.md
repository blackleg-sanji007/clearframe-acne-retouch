# Model assets

- `acne.pt`: the included YOLOv8s detector fine-tuned locally on ACNE04, approximately 22.5 MB.
- `acne.json`: architecture, provenance, test results, and SHA-256 checksum.
- `deepfillv2_WGAN.pth`: optional, separately downloaded generator checkpoint; ignored by Git.

The included detector was selected at epoch 38 in a run that completed 53 epochs. Its checkpoint metadata was made portable without changing model tensors. See the [model card](../docs/MODEL_CARD.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).

To install DeepFill, run `python download_assets.py --deepfill` using the project environment from the repository root. The app also supports Fast mode without that asset. Trained weights are reused for inference; uploading a photograph does not retrain either model.
