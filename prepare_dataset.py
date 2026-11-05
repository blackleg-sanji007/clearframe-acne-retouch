"""Validate ACNE04 Pascal VOC boxes and create deterministic YOLO splits."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_labels(xml_path, image_path):
    tree = ET.parse(xml_path).getroot()
    with Image.open(image_path) as image:
        width, height = image.size
    xml_width = int(float(tree.findtext("size/width", str(width))))
    xml_height = int(float(tree.findtext("size/height", str(height))))
    if (xml_width, xml_height) != (width, height):
        raise ValueError(f"Annotation dimensions disagree with image: {xml_path}")
    labels = []
    for obj in tree.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box: {xml_path}")
        # VOC coordinates are 1-based inclusive; YOLO boxes use pixel boundaries.
        x1 = float(box.findtext("xmin")) - 1
        y1 = float(box.findtext("ymin")) - 1
        x2 = float(box.findtext("xmax"))
        y2 = float(box.findtext("ymax"))
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(f"Non-finite box: {xml_path}")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: {xml_path}")
        labels.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(labels) + ("\n" if labels else "")


def split_samples(samples, seed=42):
    groups = defaultdict(list)
    for sample in samples:
        # Stratify by severity encoded in ACNE04 filenames.
        match = re.search(r"lev(?:le|el)[_-]?(\d)", sample[0].stem, re.I)
        groups[match.group(1) if match else "other"].append(sample)
    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for severity in sorted(groups):
        group = sorted(groups[severity], key=lambda item: item[0].name.lower())
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n*0.1)) if n >= 3 else 0
        n_val = max(1, round(n*0.2)) if n >= 3 else 0
        splits["test"].extend(group[:n_test])
        splits["val"].extend(group[n_test:n_test+n_val])
        splits["train"].extend(group[n_test+n_val:])
    if any(not items for items in splits.values()):
        raise ValueError("Need enough annotated images for nonempty train, validation and test splits.")
    return splits


def prepare(source, output, seed=42):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f"Source folder missing: {source}")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Reuse data.yaml, or choose a new --output folder.")
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the raw data folder.")
    images = defaultdict(list)
    for path in source.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.stem.lower()].append(path)
    annotations = sorted(source.rglob("*.xml"))
    if not annotations:
        raise ValueError("No Pascal VOC XML annotations found. Extract Detection.tar first.")
    samples, names, hashes = [], set(), set()
    skipped_duplicates = 0
    for xml_path in annotations:
        matches = images.get(xml_path.stem.lower(), [])
        if not matches:
            raise ValueError(f"No image for {xml_path.name}. Extract Classification.tar too.")
        digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}
        if len(digests) != 1:
            raise ValueError(f"Different photos share one annotation name: {xml_path.stem}")
        image_path, digest = sorted(matches)[0], next(iter(digests))
        name = image_path.stem.lower()
        if name in names or digest in hashes:
            skipped_duplicates += 1
            continue
        labels = voc_labels(xml_path, image_path)
        names.add(name)
        hashes.add(digest)
        samples.append((image_path, labels, digest))
    splits = split_samples(samples, seed)
    manifest = {"seed": seed, "split_type": "image-level severity-stratified 70/20/10", "source": str(source), "duplicates_removed": skipped_duplicates, "splits": {}}
    for split, items in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        manifest["splits"][split] = []
        for image_path, labels, digest in items:
            shutil.copy2(image_path, image_dir / image_path.name)
            (label_dir / f"{image_path.stem}.txt").write_text(labels, encoding="utf-8")
            manifest["splits"][split].append({"image": image_path.name, "sha256": digest})
    config = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "pimple"}}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(samples)} photos: " + ", ".join(f"{key}={len(value)}" for key, value in splits.items()))
    print(f"Removed {skipped_duplicates} exact duplicate samples. Config: {output / 'data.yaml'}")
    print("Image-level split: patient identities are unavailable; patient independence is not established.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/acne04")
    parser.add_argument("--output", type=Path, default=ROOT / "data/acne04")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
