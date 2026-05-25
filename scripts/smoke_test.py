"""Tiny GPU/PyTorch smoke test for Miyabi container."""
import platform
import sys

print("== Python/platform ==")
print("python:", sys.version.split()[0])
print("machine:", platform.machine())

print("\n== Torch ==")
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  mem={p.total_memory/1e9:.1f}GB  cc={p.major}.{p.minor}")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul ok, output norm:", float(y.norm()))
