# Experiments and measured results

[Back to the README](../README.md)

## Scope of the experiment

This experiment fine-tunes a single-class acne lesion detector and evaluates it on held-out images. It also checks that both restoration methods work and preserve pixels outside the selected mask. Detection accuracy and restoration realism are separate questions; the reported mAP values measure only detection.

## Dataset preparation

Source: [ACNE04, provided by the authors of Joint Acne Image Grading and Counting via Label Distribution Learning](https://github.com/xpwu95/LDL). Their repository specifies academic use.

| Preparation step | Count / setting |
| --- | ---: |
| Annotated source samples processed | 1,457 |
| Exact-image duplicate samples removed | 38 |
| Unique images retained | 1,419 |
| Training images | 993 |
| Validation images | 283 |
| Test images | 143 |
| Test lesion annotations | 1,859 |
| Split seed | 42 |

The script validates XML/image dimensions and finite, nonempty boxes, applies the Pascal VOC coordinate convention, and converts boxes into normalized YOLO coordinates. SHA-256 hashes detect byte-identical photographs. The split is stratified using the severity marker in filenames; severity is used only for splitting, not as a prediction target.

The [manifest](../reports/split_manifest.json) records filenames and content hashes for each split without including photographs or the original machine's paths. Exact hashes do not identify differently compressed or cropped copies. Near-duplicate filtering was not performed. Patient identifiers are unavailable, so the split is not established to be patient-independent.

## Training configuration

| Setting | Value |
| --- | --- |
| Architecture | YOLOv8s, single lesion class |
| Starting weights | General COCO-pretrained `yolov8s.pt` |
| Ultralytics version | 8.3.228 |
| PyTorch version | 2.7.1+cu128 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM |
| System RAM | 16 GB |
| Image-size target | 640 px |
| Batch size / loader workers | 8 / 0 |
| Optimizer / initial learning rate | AdamW / 0.001 |
| Mixed precision | Enabled |
| RAM image cache | Disabled |
| Target epochs / early-stopping patience | 60 / 15 |
| Completed epochs / selected epoch | 53 / 38 |
| Model selection fitness | Validation mAP@0.50:0.95 in the pinned implementation |

Augmentation uses horizontal flipping, rotation up to 8 degrees, translation 0.05, scale 0.25, moderate HSV changes, and mosaic probability 0.5, with mosaic closed for the final 10 target epochs. Vertical flipping and mixup are disabled. These choices limit implausible facial transforms while retaining variation in position, lighting, and scale.

Batch 8 and mixed precision fit this laptop. Observed training GPU allocations varied roughly from 2.0 to 2.7 GB; this is not a guarantee for another environment or a full process-memory measurement. Multiple Windows data-loader workers caused system-memory pressure in preliminary runs, so the recorded run uses workers 0. The trial was not a systematic speed benchmark.

The run was resumed after interruptions, including a Windows restart. Checkpoints restored optimizer state, and the old optional time cap was removed. Because resuming restarts parts of the data-loading/random state, a new uninterrupted run should not be expected to reproduce every numeric result exactly.

The per-epoch [training CSV](../reports/training.csv) contains 53 rows. Its `time` column restarts after resumption; it is not one cumulative wall-clock measurement. The learning-curve plot uses epoch numbers rather than this column.

## Detector evaluation

The best validation checkpoint, from epoch 38, was evaluated on the 143-image test split. Evaluation uses the pinned library's default matching and threshold-sweep procedures unless overridden in [evaluate.py](../evaluate.py).

| Test metric | Final result |
| --- | ---: |
| Precision | 0.469746 |
| Recall | 0.461292 |
| mAP@0.50 | 0.385953 |
| mAP@0.50:0.95 | 0.099081 |

Full-precision values and the published checkpoint checksum are in [test_metrics.json](../reports/test_metrics.json).

IoU measures box overlap. AP summarizes a precision–recall curve at a specified IoU threshold. mAP@0.50 uses IoU 0.50; mAP@0.50:0.95 averages ten thresholds from 0.50 through 0.95. With one class, the class-average AP is that class's AP. Precision and recall are the operating-point summaries returned by Ultralytics; they should not be interpreted as the app's fixed-confidence 0.25 results.

The gap between mAP@0.50 and the stricter averaged score indicates sensitivity to precise localization. The model also misses lesions and produces false positives. No claim of clinical reliability follows from these numbers.

### Comparison with the initial checkpoint

The early two-epoch checkpoint produced test mAP@0.50 **0.2629** and mAP@0.50:0.95 **0.0638**. The completed run improved those figures to **0.3860** and **0.0991**. This comparison shows improvement over this project's own early baseline, not superiority to the original course project, other papers, or other detectors.

The baseline test evaluation was already observed before the final evaluation. The test images were not included in training or used for checkpoint selection; no test-driven hyperparameter search is claimed.

## Restoration checks and latency

A manual app API smoke check used one public dataset photograph at confidence 0.15, selected all 10 retained boxes, and ran both restoration modes with padding 2 and strength 1.0.

| Operation | Warm sample time |
| --- | ---: |
| Detector request | 0.17 s |
| Fast (OpenCV) restoration | 0.17 s |
| DeepFillv2 restoration | 0.69 s |

Startup preparation is excluded. These are pipeline timings from one request per operation, not end-to-end browser latency or dataset-wide percentiles. Different images, masks, GPU power settings, and background workloads can change the results. OpenCV restoration itself runs on the CPU.

Both restoration paths passed an exact array-equality check outside the mask. Empty selection also preserved the whole working image. Eight local unit tests passed, including a Windows status-file sharing regression. The optional running-app check exercises the actual detector, both restoration methods, and the review state. The browser layout was not visually validated with browser automation.

## What was not measured

- No clean-face paired targets were available for valid restoration PSNR or SSIM measurements.
- No blinded human preference study or validated perceptual-quality benchmark was run.
- No controlled comparison across detector sizes, other detectors, masks, or training schedules was run.
- No patient-independent, skin-tone subgroup, age subgroup, or external-dataset evaluation was established.
- CPU latency, concurrent-user throughput, and deployment performance were not benchmarked.

DeepFill looking more natural on a selected example is therefore a qualitative observation. Its gated generator and contextual attention provide an explanation for that behavior, not evidence of universal superiority over Telea.

## Evidence and reproducibility

- [Per-epoch metrics](../reports/training.csv)
- [Image split and hashes](../reports/split_manifest.json)
- [Test results](../reports/test_metrics.json)
- [Portable checkpoint metadata](../models/acne.json)
- [Setup and reproduction instructions](REPRODUCIBILITY.md)
