# Algorithm and design rationale

[Back to the README](../README.md)

Clearframe separates lesion localization from image completion. A detector predicts candidate regions. A user chooses which ones to change. An inpainting method fills those regions, and a common compositor controls the final edit.

This separation matters: a strong generator cannot correct an incorrect mask. A convincing fill in a box over an eye is still an incorrect edit.

## 1. Normalize the photograph

`normalize_image` in [pipeline.py](../acne_demo/pipeline.py) applies EXIF orientation, converts the image to RGB, and resizes it proportionally to a maximum longest side of 1600 px. This bounds processing and interface memory without stretching the face.

All subsequent boxes and mask padding refer to this working image. The downloaded result has its resolution. Exact preservation outside the mask refers to these normalized pixels, not the original full-resolution file or its metadata.

## 2. Detect candidate lesions with YOLOv8s

### What the detector does

YOLO is a single-stage object detector: a forward pass predicts boxes and class scores at multiple feature scales. Here its only class is `pimple`. Each retained detection contains `(x1, y1, x2, y2, confidence, class)`.

The YOLOv8 backbone extracts image features. Its neck combines features across scales, and its anchor-free detection head predicts box coordinates and class scores. Multi-scale features help represent objects of different sizes, though small lesions remain difficult after resizing.

The app calls the detector with `imgsz=640`, the selected confidence threshold, non-maximum suppression IoU `0.4`, and a maximum of 500 detections. Ultralytics handles inference resizing/letterboxing and maps the resulting boxes back into working-image coordinates. Retained boxes are sorted from top to bottom for stable review numbers.

### How it was trained

The starting point was general COCO-pretrained `yolov8s.pt`. Fine-tuning updates its parameters against ACNE04 lesion annotations. We deliberately did not initialize from the original course project's acne checkpoint, whose prior exposure to these images is unknown.

The pinned Ultralytics implementation combines classification binary cross-entropy, a Complete IoU box loss, and distribution focal loss for box regression. These losses respectively supervise the lesion class and the location/shape of predicted boxes. AdamW updates the parameters; mixed precision reduces training memory and arithmetic cost on the GPU.

This is **object detection**, not pixel segmentation. The data preparation combines annotated lesions into one class. No lesion subtype classifier, face identity model, or severity grading model is trained here.

### Why the small model

YOLOv8s retained a useful model capacity while fitting the demonstrated 6 GB GPU. A smaller model could reduce memory and computation; a larger one could increase capacity and cost. No local model-size comparison was run, so this is a practical hardware choice rather than an experimentally proven optimum.

Reference: [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/). The implementation is pinned to Ultralytics 8.3.228.

## 3. Review boxes and construct the edit mask

Every detection initially appears selected. The user can deselect false positives before restoration. The UI also invalidates old detections when the image or confidence changes, and verifies that the current working image matches the detection result before applying an edit.

For each selected box, `make_mask`:

1. Floors its minimum coordinates and ceils its maximum coordinates.
2. Expands it by the requested integer padding, default 2 px.
3. Clips the rectangle to the image boundaries.
4. Sets pixels in that rectangle to 255; all other mask pixels remain 0.

Overlapping rectangles form one union mask. Rectangles match the available box annotations and cover the full predicted region. They also include non-lesion pixels inside each box. A future lesion segmentation model could provide more precise coverage, but this version does not have segmentation labels or such a model.

Small masks are preferable: extra padding may remove surrounding redness, but it also asks the restoration method to replace more genuine skin. An empty selection returns the working image unchanged without loading a restoration model.

## 4A. DeepFillv2: learned image completion

### What runs in this repository

The adapter in [inpainting.py](../acne_demo/inpainting.py) loads a pretrained `GatedGenerator` from the separately downloaded [PyTorch port](https://github.com/vrindaprabhu/deepfillv2_colab/tree/74c4e0c722ee0c017c29bf2ab422766ee5b8b746). Its architecture has a coarse stage, a refinement stage with convolution and contextual-attention branches, and a decoder that combines their features.

Only this generator runs during photo editing. There is no discriminator or training loop in an upload request. The checkpoint is used as supplied; its exact original training corpus and training recipe are not independently established here. The `WGAN` filename is not evidence that this project reproduced any particular adversarial training setup.

### Coarse completion

The input consists of the RGB image plus a binary mask channel. In the pinned implementation, masked pixels are replaced with a constant before the coarse pass. The coarse encoder/decoder proposes an initial fill; dilated convolutions expand the receptive field without continually downsampling the features.

### Gated convolution

A regular convolution treats the incoming feature map uniformly. A gated convolution learns both feature content and a gate that modulates it:

$$F = \phi(W_f * X + b_f), \qquad G = \sigma(W_g * X + b_g)$$

$$Y = F \odot G.$$

Here `*` is convolution, `phi` is the feature activation, `sigma` is sigmoid, and the multiplication is elementwise. The gate can suppress unhelpful features and pass useful ones at each spatial position and channel. It is learned from data; it is not simply the original binary edit mask.

### Refinement and contextual attention

The coarse fill is composited into the missing area, then refined. One branch processes it with convolutions. Another compares feature patches with visible-image context, assigns similarity-based attention weights, and uses the corresponding contextual features to support completion. The branches are combined and decoded to RGB.

Contextual attention operates on learned features rather than a literal copy-and-paste selection of raw skin pixels. It helps use information beyond the immediate boundary, but it does not guarantee anatomically correct or faithful reconstruction.

### The 512 px working canvas

The adapter scales the image and mask proportionally into a 512 × 512 canvas. It pads the image by replicating the border and pads the mask with zeros. The mask uses nearest-neighbor resizing to remain binary. The input follows the pinned port's `[0, 1]` preprocessing convention; the refined output is clamped to that range, unpadded, and resized back.

Bounding this canvas keeps contextual-attention memory practical on the laptop. The cost is reduced fine detail, and sufficiently tiny mask regions may be lost when the mask is downsampled. This is one reason DeepFill can fail to improve a small spot.

References: [gated convolution paper](https://arxiv.org/abs/1806.03589), [contextual attention paper](https://arxiv.org/abs/1801.07892), and the pinned port's `network.py` and `network_module.py`. The paper and the port should not be treated as identical training implementations.

## 4B. Fast mode: OpenCV Telea inpainting

The fast branch calls:

```python
cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
```

RGB is converted to BGR for OpenCV and converted back afterward. The radius is 3 pixels. This is specifically **Telea's algorithm**, not OpenCV's alternative Navier–Stokes mode.

Telea uses a fast marching strategy to progress from the known boundary into the unknown region. A pixel near that boundary is estimated from nearby known values and local directional information, with weighting that accounts for distance and the fill front. Newly filled pixels allow the process to advance farther inward.

It requires no learned restoration parameters and runs through OpenCV on the CPU. Its local estimates can blend small defects into a smooth neighborhood efficiently. As masks grow or cross detailed texture, propagating nearby information can smear texture or flatten the repaired area.

Reference: [OpenCV's inpainting explanation and Telea citation](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html).

## 5. Blend the proposal inside the mask

Both methods produce a proposed filled image `F`. The final step uses the original working image `I`, a binary selection mask `M`, and strength `s` between 0 and 1.

For each selected pixel, a distance transform estimates its distance `d` from the unselected region. An inward feather controls the blend:

$$\alpha(x,y) = s\,\min\left(\frac{d(x,y)}{3},1\right).$$

$$O(x,y) = \operatorname{round}\left[(1-\alpha(x,y))I(x,y)+\alpha(x,y)F(x,y)\right].$$

Values are clipped to the 8-bit range. Pixels with `M=0` are then explicitly copied from `I`, ensuring exact equality outside the mask. The distance transform uses OpenCV's approximate L2 transform with a 3 × 3 neighborhood, so the feather is approximately three working-image pixels wide.

The blend never extends outward from the mask. Strength 0 returns the original image; strength 1 uses the proposal fully in mask interiors while retaining the boundary transition.

## Why DeepFill can look better than Fast

The two methods answer different completion problems. Telea estimates missing colors from a local neighborhood. DeepFill also has a learned prior and access to broader feature context. When a mask removes textured skin, a learned completion can look more natural than a smooth local interpolation. That is the plausible explanation for a visual preference for DeepFill in this demo.

It is not a proven general result. Very small masks in uniform skin often suit Telea well. DeepFill can introduce false detail, inconsistent lighting, or low-resolution artifacts. Both methods depend on correct masks. There are no paired clean targets, realism scores, or controlled preference-test results in this repository.

## Why new uploads are fast

Training performs repeated forward/backward passes and optimizer updates over the dataset. It is a separate offline command. Uploads perform forward inference using saved parameters.

`AcnePipeline` retains the detector and restoration generator in memory. Startup warmup on CUDA initializes them before the interface becomes available. A lock and a shared UI queue serialize model operations, preventing overlapping requests from multiplying GPU workloads. No upload starts a training job, reloads checkpoints unnecessarily, or writes shared intermediate inference files.
