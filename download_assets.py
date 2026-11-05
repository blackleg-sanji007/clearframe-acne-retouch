"""One-time downloads from the dataset authors and the upstream project."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
DATASET_SOURCE = "https://github.com/xpwu95/LDL"
DEEPFILL_COMMIT = "74c4e0c722ee0c017c29bf2ab422766ee5b8b746"
DEEPFILL_SHA256 = "60d338fa0c7556db043052f1d05d964c9b817687990676096af32dbfc1c48e22"


def download(file_id, destination, expected_sha256=None):
    import gdown
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        verify(destination, expected_sha256)
        print(f"Already downloaded: {destination.name}", flush=True)
        return
    part = destination.with_suffix(destination.suffix + ".part")
    result = gdown.download(id=file_id, output=str(part), quiet=False, resume=True)
    if result is None:
        raise RuntimeError(f"Download failed: {destination.name}")
    verify(part, expected_sha256)
    part.replace(destination)


def verify(path, expected_sha256):
    if expected_sha256:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {path}; remove that file and download again.")


def extract(archive, destination):
    marker = destination / f".{archive.stem}-extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} ...", flush=True)
    with tarfile.open(archive) as package:
        package.extractall(destination, filter="data")
    marker.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--deepfill", action="store_true")
    args = parser.parse_args()
    if not (args.dataset or args.deepfill):
        parser.error("Choose --dataset, --deepfill, or both.")
    if args.deepfill:
        source = ROOT / "vendor/deepfillv2"
        if not source.exists():
            subprocess.run(["git", "clone", "https://github.com/vrindaprabhu/deepfillv2_colab.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", DEEPFILL_COMMIT], check=True)
        download("1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-", ROOT / "models/deepfillv2_WGAN.pth", DEEPFILL_SHA256)
    if args.dataset:
        print(f"ACNE04, academic-use dataset. Source: {DATASET_SOURCE}", flush=True)
        # Classification.tar contains the photographs; Detection.tar contains XML boxes.
        for file_id, filename in [
            ("1NqsL5mlI3iTsPmjvVX7y6GUileM7oL5v", "Detection.tar"),
            ("1ysutxP3IHKhQA8brdwPlhwbduo-kYt1x", "Classification.tar"),
        ]:
            archive = ROOT / "data/downloads" / filename
            download(file_id, archive)
            extract(archive, ROOT / "data/raw/acne04")
        print("Dataset ready. Next: python prepare_dataset.py", flush=True)


if __name__ == "__main__":
    main()
