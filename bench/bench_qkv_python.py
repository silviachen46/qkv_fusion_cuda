# bench/bench_qkv_python.py
import time, torch
from torch_integration.fused_qkv_python import fused_qkv_python


@torch.no_grad()
def main(B=4, H=4096, n_heads=32, iters=50):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(B, H, device=device)
    W_qkv = torch.randn(H, 3*H, device=device)
    b_qkv = torch.randn(3*H, device=device)

    # warmup
    for _ in range(5):
        fused_qkv_python(x, W_qkv, b_qkv, n_heads)

    t0 = time.time()
    for _ in range(iters):
        fused_qkv_python(x, W_qkv, b_qkv, n_heads)
    if device == "mps": torch.mps.synchronize()
    t1 = time.time()
    print(f"avg {(t1-t0)/iters*1000:.3f} ms  (device={device})")

if __name__ == "__main__":
    main()
