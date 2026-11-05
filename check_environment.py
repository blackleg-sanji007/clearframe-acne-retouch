"""Check imports and perform a small real CUDA operation."""
import importlib.metadata
import platform


def main():
    import torch
    print(f"Python {platform.python_version()}")
    for name in ["torch", "torchvision", "ultralytics", "gradio", "opencv-python"]:
        print(f"{name}: {importlib.metadata.version(name)}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        x = torch.randn(256, 256, device="cuda")
        result = (x @ x.T).mean().item()
        print(f"GPU: {props.name}; VRAM: {props.total_memory/1024**3:.1f} GB")
        print(f"CUDA matrix operation passed: {result:.4f}")
    else:
        print("CPU inference is possible. For GPU training, run setup.ps1 to install the CUDA build.")


if __name__ == "__main__":
    main()
