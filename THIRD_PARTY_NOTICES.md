# Third-party sources and acknowledgments

Clearframe's project license covers its source code. It does not replace the terms attached to externally maintained libraries, datasets, implementations, or pretrained models.

## Original project inspiration

[architk22/EE604-Course-Project](https://github.com/architk22/EE604-Course-Project) inspired the detection-plus-inpainting direction. The public Clearframe repository starts with a fresh Git history and does not include that repository's notebooks, paper PDF, `best_final.pt`, helper scripts, README content, or original application file.

The new `demo.py` interface, reusable local pipeline, dataset preparation, training/evaluation tools, tests, and documentation were prepared for this remake. This acknowledgment credits the project direction without representing YOLO, DeepFill, or Telea as new algorithms developed here.

## Detection

- **Ultralytics YOLOv8**, pinned package 8.3.228: [source](https://github.com/ultralytics/ultralytics/tree/v8.3.228), [documentation](https://docs.ultralytics.com/models/yolov8/), [license](https://github.com/ultralytics/ultralytics/blob/v8.3.228/LICENSE).
- Training initialized from general COCO-pretrained `yolov8s.pt`. The included `models/acne.pt` contains the detector fine-tuned locally for this project; it is not an upstream acne checkpoint.
- Ultralytics publishes AGPL-3.0 and separate enterprise licensing options. Its licensing terms remain applicable; this project does not claim unrestricted rights over upstream implementation or pretrained assets.

## DeepFill research and implementation

Research:

- Yu et al., **Generative Image Inpainting with Contextual Attention**, CVPR 2018. [Paper](https://arxiv.org/abs/1801.07892).
- Yu et al., **Free-Form Image Inpainting with Gated Convolution**, ICCV 2019. [Paper](https://arxiv.org/abs/1806.03589).
- Authors' [generative inpainting repository](https://github.com/JiahuiYu/generative_inpainting).

Runtime implementation:

- [vrindaprabhu/deepfillv2_colab](https://github.com/vrindaprabhu/deepfillv2_colab), pinned to `74c4e0c722ee0c017c29bf2ab422766ee5b8b746`.
- That port credits [csqiangwen/DeepFillv2_Pytorch](https://github.com/csqiangwen/DeepFillv2_Pytorch).
- Generator checkpoint: [separate Google Drive download](https://drive.google.com/file/d/1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-/view), SHA-256 `60d338fa0c7556db043052f1d05d964c9b817687990676096af32dbfc1c48e22`.

The downloaded source and weights are not committed to Clearframe. They are obtained only through the explicit asset download command and retain their upstream terms. A redistribution license for that particular port/checkpoint is not asserted here; check with its maintainers before redistributing it. The original checkpoint's exact training corpus and loss recipe were not independently verified, and no DeepFill training was performed for this remake.

## Dataset

**ACNE04**, from the authors of *Joint Acne Image Grading and Counting via Label Distribution Learning*, ICCV 2019: [authors' repository and citation](https://github.com/xpwu95/LDL), [download folder](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ).

The authors state that the work is free for academic usage and direct other uses to the corresponding author. Photographs, XML annotations, and archives are not redistributed in this repository. The published split manifest contains filenames and content hashes for experiment reproducibility; it does not contain photographs.

The source-code license does not grant new rights over the dataset or override the authors' terms. The trained detector is provided for the local educational demonstration described in its [model card](docs/MODEL_CARD.md).

## Other runtime foundations

| Component | Use | Upstream |
| --- | --- | --- |
| PyTorch / torchvision | Tensor operations, CUDA inference and training | [PyTorch](https://github.com/pytorch/pytorch) |
| OpenCV | Telea inpainting, image operations, masks | [OpenCV](https://github.com/opencv/opencv) |
| Gradio | Local upload, review, and comparison UI | [Gradio](https://github.com/gradio-app/gradio) |
| NumPy / Pillow | Image arrays and decoding | [NumPy](https://github.com/numpy/numpy), [Pillow](https://github.com/python-pillow/Pillow) |
| Matplotlib | Training curve visualization | [Matplotlib](https://github.com/matplotlib/matplotlib) |
| PyYAML / gdown / psutil | Data configuration, downloads, job supervision | [PyYAML](https://github.com/yaml/pyyaml), [gdown](https://github.com/wkentaro/gdown), [psutil](https://github.com/giampaolo/psutil) |
| uv | Project-local Windows Python environment setup | [uv](https://github.com/astral-sh/uv) |

Each dependency retains the license and notices in its distribution. No third-party source is copied into the Clearframe Git history as a vendored library.
