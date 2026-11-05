"""Local photo retouching demo. Run with .venv/Scripts/python.exe demo.py."""
import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

import gradio as gr
import numpy as np
import torch
from acne_demo.pipeline import AcnePipeline, draw_detections, normalize_image

LOG = logging.getLogger(__name__)


def create_app(pipeline):
    def detect(image, confidence):
        try:
            detection = pipeline.detect(image, confidence)
            choices = [(f"Spot {i+1} ({box[4]:.0%})", str(i)) for i, box in enumerate(detection.boxes)]
            size = detection.rgb.shape
            message = f"Found {len(choices)} spots in {detection.elapsed:.2f} s. Deselect any incorrect detections, then apply retouch."
            if not choices:
                message = "No spots detected at this confidence. Try a lower threshold if visible spots were missed."
            if detection.original_size != (size[1], size[0]):
                message += f" Working image resized to {size[1]} × {size[0]} px."
            return detection, draw_detections(detection), gr.update(choices=choices, value=[v for _, v in choices]), message, None, None
        except Exception as error:
            LOG.exception("Detection failed")
            raise gr.Error(str(error)) from error

    def restore(detection, selected, mode, padding, strength, current_image):
        if detection is None:
            raise gr.Error("Find spots first.")
        try:
            current_rgb, _ = normalize_image(current_image)
            if not np.array_equal(current_rgb, detection.rgb):
                raise ValueError("The photo changed while detection was running. Find spots again.")
            selected = [int(value) for value in (selected or [])]
            output, mask, elapsed = pipeline.restore(detection, selected, padding, strength, mode)
            message = f"Retouched {len(selected)} selected spots with {mode} in {elapsed:.2f} s. Drag the divider to compare."
            if not selected:
                message = "No spots selected. The image is unchanged."
            return (detection.rgb, output), mask, message
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise gr.Error("GPU memory is full. Close other GPU apps or select Fast (OpenCV).") from error
        except Exception as error:
            LOG.exception("Retouch failed")
            raise gr.Error(str(error)) from error

    def reset():
        return None, None, gr.update(choices=[], value=[]), "Upload a photo, then find spots.", None, None

    source = "Acne detection checkpoint"
    with gr.Blocks(
        title="Clearframe · Local photo retouch",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        delete_cache=(3600, 3600),
        css=".gradio-container {max-width: 1180px !important;} #intro {padding: 12px 0 8px;} footer {display:none !important;}",
    ) as demo:
        gr.Markdown("# Clearframe\n### A simple acne retouching demo", elem_id="intro")
        gr.Markdown("Upload a clear face photo, review detected spots, and compare the retouch. Processing stays on this laptop.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="1 · Your photo", height=360)
                confidence = gr.Slider(0.05, 0.8, value=0.25, step=0.01, label="Detection confidence", info="Lower finds more spots, but may include facial features.")
                find = gr.Button("Find spots", variant="primary")
                examples = []
                for severity in (1, 0):
                    candidates = sorted((ROOT / "data/acne04/images/test").glob(f"levle{severity}*"))
                    if candidates:
                        examples.append([str(candidates[0])])
                if examples:
                    gr.Examples(examples=examples, inputs=[image], cache_examples=False, label="Try a dataset sample")
            with gr.Column(scale=1):
                preview = gr.Image(label="2 · Review numbered spots", height=360, interactive=False, format="png")
                selection = gr.CheckboxGroup(choices=[], label="Spots to retouch", info="Deselect spots that overlap moles, lips, eyes, or other features.")
        with gr.Row():
            preferred_mode = "DeepFillv2" if pipeline.device.startswith("cuda") and (ROOT / "models/deepfillv2_WGAN.pth").is_file() else "Fast (OpenCV)"
            mode = gr.Radio(["DeepFillv2", "Fast (OpenCV)"], value=preferred_mode, label="Restoration method", info="DeepFill generates texture; Fast uses neighboring pixels.")
            strength = gr.Slider(0, 1, value=1, step=0.05, label="Retouch strength")
        with gr.Accordion("Adjust mask coverage", open=False):
            padding = gr.Slider(0, 12, value=2, step=1, label="Extra coverage around each spot (px)", info="Keep small to preserve surrounding skin.")
        apply = gr.Button("Apply retouch", variant="primary")
        status = gr.Markdown("Upload a photo, then find spots.")
        comparison = gr.ImageSlider(label="3 · Before / after — drag to compare", interactive=False, format="png")
        with gr.Accordion("View or download the mask", open=False):
            mask = gr.Image(label="White areas are retouched", interactive=False, format="png")
        gr.Markdown(f"{source} · {pipeline.weights.name} · {pipeline.device.upper()}\n\nDemo for photo editing. Results depend on lighting and detection accuracy. Generated skin may look artificial.")
        outputs = [state, preview, selection, status, comparison, mask]
        find.click(detect, [image, confidence], outputs, concurrency_limit=1, concurrency_id="gpu", api_name="detect")
        apply.click(restore, [state, selection, mode, padding, strength, image], [comparison, mask, status], concurrency_limit=1, concurrency_id="gpu", api_name="retouch")
        image.change(reset, outputs=outputs, queue=False, api_name=False)
        confidence.input(reset, outputs=outputs, queue=False, api_name=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-warmup", action="store_true", help="Skip startup preparation; the first upload will take longer")
    args = parser.parse_args()
    pipeline = AcnePipeline(args.weights, args.device)
    print(f"Loaded {pipeline.weights.name} on {pipeline.device}.")
    if not args.no_warmup:
        print("Preparing models before accepting photos. This takes a moment...")
        has_deepfill = (ROOT / "models/deepfillv2_WGAN.pth").is_file() and (ROOT / "vendor/deepfillv2/deepfillv2/network.py").is_file()
        pipeline.warmup(include_deepfill=has_deepfill and pipeline.device.startswith("cuda"))
        print("Models are ready and will be reused for every upload.")
    create_app(pipeline).queue(max_size=4, default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=not args.no_browser, show_error=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
