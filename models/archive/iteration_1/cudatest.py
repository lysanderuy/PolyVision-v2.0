import torch

# Check 1: Is CUDA available?
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

# If it is available, print more details
if cuda_available:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")