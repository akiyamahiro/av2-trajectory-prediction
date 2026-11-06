"""Run this first. It performs a GPU computation, not just a version check."""
import sys
print("Python:", sys.version)
print("Interpreter:", sys.executable)
try:
    import torch
except ImportError:
    raise SystemExit("PyTorch missing. Use https://pytorch.org/get-started/locally/ for Linux + CUDA installation.")
print("PyTorch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(256, 256, device="cuda")
    print("GPU matrix multiplication:", (x @ x.T).mean().item())
else:
    print("CPU works, but planned A5000 training needs CUDA in this interpreter.")
for name in ("numpy", "pandas", "pyarrow", "matplotlib"):
    module = __import__(name)
    print(name, module.__version__)
