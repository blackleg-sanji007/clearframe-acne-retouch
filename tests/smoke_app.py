"""Manual end-to-end check against a running local Gradio server."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    client = Client(args.url, verbose=False)
    candidates = sorted((ROOT / "data/acne04/images/test").glob("levle1*"))
    path = args.image or (candidates[0] if candidates else None)
    if path is None or not path.is_file():
        parser.error("Provide --image PATH, or prepare the ACNE04 dataset first.")
    image = handle_file(str(path))
    result = client.predict(image, 0.15, api_name="/detect")
    print(result[2], flush=True)
    selection = result[1]["value"] if isinstance(result[1], dict) else result[1]
    assert selection, "Choose a photo with detectable spots."
    for mode in ["Fast (OpenCV)", "DeepFillv2"]:
        comparison, mask_path, status = client.predict(selection, mode, 2, 1.0, image, api_name="/retouch")
        before, after = [np.asarray(Image.open(filename).convert("RGB")) for filename in comparison]
        mask = np.asarray(Image.open(mask_path).convert("L"))
        np.testing.assert_array_equal(before[mask == 0], after[mask == 0])
        print(status, flush=True)
    comparison, _, status = client.predict([], "Fast (OpenCV)", 2, 1.0, image, api_name="/retouch")
    np.testing.assert_array_equal(np.asarray(Image.open(comparison[0])), np.asarray(Image.open(comparison[1])))
    print("Empty selection preserves the original image. API checks passed.", flush=True)


if __name__ == "__main__":
    main()
