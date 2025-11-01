# Fused QKV for LLM Decoding (CUDA + PyTorch C++ Extension)

## Overview
This project implements a **fused QKV CUDA operator** for LLM decoding.
It merges Query/Key/Value projections into a single GEMM pass with fused bias, rotary position embedding (RoPE),
and KV-cache writeback in the epilogue, reducing kernel launches and memory traffic for faster inference.

## Tech Stack
- CUDA / C++17 / CUTLASS
- PyTorch C++ Extension (pybind11)
- Nsight Systems / Nsight Compute / torch.profiler
- HuggingFace Transformers integration

## Repository Structure
```
my_ops/             - custom CUDA operators
torch_integration/  - PyTorch model integration
bench/              - benchmarks and plots
tools/              - helper scripts (weight packing etc.)
tests/              - correctness tests
results/            - metrics and graphs
```

## Example Results
| Batch | Tokens | Baseline (tok/s) | Fused (tok/s) | Speedup |
|:------|:--------|:----------------|:--------------|:--------|
| 1 | 128 | 540 | **710** | **1.31×** |
| 4 | 256 | 1880 | **2500** | **1.33×** |

GPU: A100 40GB  Precision: FP16  
Kernel launches ↓ 40%  Memory peak ↓ 18%

## How to Run
```bash
python3 -m venv qkv_env && source qkv_env/bin/activate
pip install torch torchvision transformers accelerate matplotlib tqdm
python setup.py build_ext --inplace
python tools/pack_qkv.py --model mistral-7b --out weights/qkv_packed.pt
python bench/bench_decode_cuda.py --model mistral-7b --batch 4 --new_tokens 256 --fusion off
python bench/bench_decode_cuda.py --model mistral-7b --batch 4 --new_tokens 256 --fusion on
```

## Key Features
- Single-pass QKV GEMM (fused bias + RoPE + KV-cache writeback)
- CUTLASS-based tiling optimization
- Automated benchmark & profiling
- Integration with HuggingFace transformers

## License
MIT License © 2025 Silvia Chen
