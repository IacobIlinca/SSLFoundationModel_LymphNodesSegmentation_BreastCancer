import torch, time
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Allocate ~1GB (adjust down if GPU is small/busy)
x = torch.randn(16384, 16384, device="cuda")  # ~1.0 GB
print("Allocated, sleeping 60s... check nvidia-smi now.")
time.sleep(60)