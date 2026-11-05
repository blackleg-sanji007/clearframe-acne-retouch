# Model card: Clearframe acne detector

[Back to the README](../README.md)

## Model identity

| Field | Value |
| --- | --- |
| Checkpoint | `models/acne.pt` |
| Architecture | Ultralytics YOLOv8s |
| Task | Single-class bounding-box detection |
| Class | `pimple` |
| Initialization | COCO-pretrained `yolov8s.pt` |
| Fine-tuning dataset | ACNE04 |
| Training | Local RTX 3050 Laptop GPU, 6 GB VRAM |
| Completed / best epoch | 53 / 38 |
| Selection criterion | Validation mAP@0.50:0.95 |
| Checkpoint format | Ultralytics/PyTorch, optimizer stripped |
| Metadata and checksum | [models/acne.json](../models/acne.json) |

The detector weights were trained for this remake. They are not the `best_final.pt` checkpoint from the original course repository. Publication removed machine-specific path and source-checkout Git metadata while verifying that every model tensor remained unchanged.

## Intended use

Local educational demonstrations of detection, reviewable masks, and selective image inpainting. The detector identifies candidate image regions; a person reviews the selections before retouching.

It is not intended to diagnose acne, recommend treatment, determine severity, identify a person, or make clinical or demographic judgments. The one-class labels do not support lesion subtype claims.

## Data and evaluation

The experiment used 1,419 unique images after removing 38 byte-identical duplicates, split into 993 training, 283 validation, and 143 test images. Splitting used seed 42 and severity markers in filenames. Images and annotations are downloaded separately from the [dataset authors](https://github.com/xpwu95/LDL), who specify academic usage.

Test results over 1,859 annotated lesions: precision **0.469746**, recall **0.461292**, mAP@0.50 **0.385953**, and mAP@0.50:0.95 **0.099081**. See [EXPERIMENTS.md](EXPERIMENTS.md) for the metric definitions, baseline comparison, and limitations.

## Limitations

- Small lesions, lighting changes, occlusion, image resizing, and skin texture can affect detections.
- False positives may cover normal skin, moles, eyes, lips, or other facial features. Review is required for a deliberate edit.
- Boxes are not lesion segmentation masks and include some surrounding skin.
- Exact-image deduplication does not eliminate near-duplicates or establish patient independence.
- Skin-tone, age, patient-level, and external-dataset performance were not established.
- The published test results apply to this image split and checkpoint, not every photograph or a newly fine-tuned model.

## Relationship to restoration

DeepFillv2 is a separate pretrained generator, downloaded explicitly and not fine-tuned here. OpenCV Telea is a non-neural alternative. Neither changes the detector's trained parameters during an upload. Their results are composited only inside the user-selected mask.

Generated skin is a plausible visual replacement rather than a verified reconstruction of the person's unobserved skin. No restoration-quality benchmark or demographic fairness study is available in this project.

## Distribution and terms

The project source uses AGPL-3.0. Ultralytics' own terms apply to its implementation and derived model use. ACNE04 retains the dataset authors' academic-use terms; the project license does not relicense the photographs. External DeepFill source and weights retain their own terms and are not bundled. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
