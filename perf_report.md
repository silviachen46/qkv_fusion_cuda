# Performance Report (Template)

## Environment
- GPU: A100 40GB
- CUDA: 12.1
- PyTorch: 2.x (cu121)
- Nsight Systems/Compute versions:

## Results Summary
- Decoding throughput: +XX%
- Kernel launches: -YY%
- Peak memory: -ZZ%

## Nsight Snapshots
(Place timeline screenshots and kernel hotspot tables here)

## Ablation
- Only GEMM+bias+split
- + RoPE
- + KV writeback
