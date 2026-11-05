"""In-memory inpainting. Pixels outside the chosen mask are preserved exactly."""
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class DeepFill:
    def __init__(self, device: str):
        weights = ROOT / "models" / "deepfillv2_WGAN.pth"
        source = ROOT / "vendor" / "deepfillv2"
        if not weights.is_file() or not (source / "deepfillv2" / "network.py").is_file():
            raise FileNotFoundError(
                "DeepFill is not installed. Run .venv\\Scripts\\python.exe "
                "download_assets.py --deepfill, or select Fast (OpenCV)."
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from deepfillv2.network import GatedGenerator

        self.device = device
        options = SimpleNamespace(
            pad_type="zero", in_channels=4, out_channels=3,
            latent_channels=48, activation="elu", norm="in",
            use_cuda=device.startswith("cuda"),
        )
        self.model = GatedGenerator(options)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Preserve aspect ratio; the original implementation stretches to a square.
        # Bound attention memory on the 6 GB GPU using a 512 x 512 working canvas.
        h, w = mask.shape
        scale = 512 / max(h, w)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (512 - rw) // 2, (512 - rh) // 2
        image = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
        image = cv2.copyMakeBorder(image, top, 512-rh-top, left, 512-rw-left, cv2.BORDER_REPLICATE)
        small_mask = cv2.copyMakeBorder(small_mask, top, 512-rh-top, left, 512-rw-left, cv2.BORDER_CONSTANT)
        x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div_(255).to(self.device)
        m = torch.from_numpy(small_mask.copy())[None, None].float().div_(255).to(self.device)
        _, prediction = self.model(x, m)
        output = prediction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        output = np.rint(output * 255).astype(np.uint8)
        output = output[top:top+rh, left:left+rw]
        return cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)


def blend_inside_mask(rgb, filled, mask, strength=1.0):
    """Feather inward, so unselected skin and facial features cannot change."""
    if not 0 <= strength <= 1:
        raise ValueError("Retouch strength must be between 0 and 1.")
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.minimum(distance / 3.0, 1.0)[..., None] * strength
    output = np.rint(rgb * (1-alpha) + filled * alpha).clip(0, 255).astype(np.uint8)
    output[mask == 0] = rgb[mask == 0]
    return output


def fast_inpaint(rgb, mask):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
