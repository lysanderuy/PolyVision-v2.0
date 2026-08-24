import torch
import detectron2

# Check NVCC version (this will likely fail on your system)
import subprocess
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    print("NVCC Version:")
    print(result.stdout)
except FileNotFoundError:
    print("NVCC not found - using CPU version")

# Check versions
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Additional CPU confirmation
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())