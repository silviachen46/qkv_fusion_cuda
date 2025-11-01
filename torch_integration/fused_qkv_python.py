# torch_integration/fused_qkv_python.py
import torch
from typing import Tuple, Optional

@torch.no_grad()
def fused_qkv_python(
    x: torch.Tensor,          # [B, H]
    W_qkv: torch.Tensor,      # [H, 3H]
    b_qkv: torch.Tensor,      # [3H]
    n_heads: int,
    rope: Optional[dict] = None,
    kv_cache: Optional[dict] = None,
    layer_id: int = 0,
    token_pos: Optional[int] = None,
):
    B, H = x.shape
    y = x @ W_qkv            # [B, 3H]
    y = y + b_qkv            # broadcast

    q, k, v = y[:, :H], y[:, H:2*H], y[:, 2*H:]
    head_dim = H // n_heads
    q = q.view(B, n_heads, head_dim)
    k = k.view(B, n_heads, head_dim)
    v = v.view(B, n_heads, head_dim)

    # (可选) RoPE / KV 写回：先跳过，等对齐后再加
    return q, k, v
