"""Measure both modes on a public dataset photo; save a comparison locally."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch

from acne_demo.pipeline import AcnePipeline, draw_detections

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()
    image_path = args.image or sorted((ROOT / "data/acne04/images/test").glob("levle1*"))[0]
    pipeline = AcnePipeline()
    with Image.open(image_path) as image:
        detection = pipeline.detect(image, confidence=args.confidence)
    assert detection.boxes, "Choose a photo with detectable spots for this smoke check."
    output_dir = ROOT / ".cache/inference-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"image": image_path.name, "device": pipeline.device, "spots": len(detection.boxes), "detection_seconds": detection.elapsed}
    images = [("Original", detection.rgb), ("Detected spots", draw_detections(detection))]
    for mode in ["Fast (OpenCV)", "DeepFillv2"]:
        for pass_number in range(2):
            torch.cuda.reset_peak_memory_stats() if pipeline.device.startswith("cuda") else None
            output, mask, elapsed = pipeline.restore(detection, mode=mode)
            assert output.shape == detection.rgb.shape
            np.testing.assert_array_equal(output[mask == 0], detection.rgb[mask == 0])
            stats[f"{mode}_pass{pass_number+1}_seconds"] = elapsed
            if pipeline.device.startswith("cuda"):
                stats[f"{mode}_peak_gpu_mb"] = round(torch.cuda.max_memory_allocated()/1024**2, 1)
        Image.fromarray(output).save(output_dir / f"{mode.split()[0]}.png")
        images.append((mode, output))
    Image.fromarray(mask).save(output_dir / "mask.png")
    width = 360
    height = round(detection.rgb.shape[0] * width / detection.rgb.shape[1])
    grid = Image.new("RGB", (width*len(images), height+40), "#f0f3f5")
    draw = ImageDraw.Draw(grid)
    for i, (title, array) in enumerate(images):
        tile = Image.fromarray(array).resize((width, height))
        grid.paste(tile, (i*width, 40))
        draw.text((i*width+12, 12), title, fill="#223344")
    grid.save(output_dir / "comparison.jpg")
    (output_dir / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Comparison saved to {output_dir / 'comparison.jpg'}", flush=True)


if __name__ == "__main__":
    main()
