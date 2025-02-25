# CUDA test
import torch

print("PyTorch version:", torch.__version__)
print("CUDA version in PyTorch:", torch.version.cuda) 
print("CUDA available:", torch.cuda.is_available()) 
print("Number of GPUs:", torch.cuda.device_count())  
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

