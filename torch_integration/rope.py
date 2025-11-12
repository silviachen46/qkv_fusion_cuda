import torch
from typing import Optional, Tuple

@torch.no_grad()
def _cos_sin_from_positions(
    head_dim: int,
    positions: torch.Tensor,                # [...], int/float
    *,
    base: float = 10000.0,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=compute_dtype) / half))
    freqs = (positions.to(compute_dtype) * (scale * inv_freq))  # [..., half]
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)  # [..., d]
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)  # [..., d]
    return cos, sin

@torch.no_grad()
def precompute_rope(
    head_dim: int,
    *,
    pos: Optional[int] = None,                # 单步
    pos_b: Optional[torch.Tensor] = None,     # [B]
    seq_len: Optional[int] = None,            # prefill
    base: float = 10000.0,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    assert sum(x is not None for x in (pos, pos_b, seq_len)) == 1, "choose one of pos/pos_b/seq_len"
    compute_dtype = torch.float32

    if pos is not None:
        positions = torch.tensor([pos], device=device, dtype=torch.long)  # [1]
        cos, sin = _cos_sin_from_positions(head_dim, positions, base=base, scale=scale,
                                           device=device, compute_dtype=compute_dtype)  # [1,d]
        cos = cos[None, ...]  # [1,1,d]
        sin = sin[None, ...]
    elif pos_b is not None:
        assert pos_b.dim() == 1
        positions = pos_b.to(device=device)
        cos, sin = _cos_sin_from_positions(head_dim, positions, base=base, scale=scale,
                                           device=device, compute_dtype=compute_dtype)  # [B,d]
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    else:
        T = int(seq_len)
        positions = torch.arange(T, device=device, dtype=torch.long)  # [T]
        cos, sin = _cos_sin_from_positions(head_dim, positions, base=base, scale=scale,
                                           device=device, compute_dtype=compute_dtype)  # [T,d]
        cos = cos[None, None, ...]  # [1,1,T,d]
        sin = sin[None, None, ...]
    if out_dtype is not None:
        cos = cos.to(out_dtype)
        sin = sin.to(out_dtype)
    return cos, sin

@torch.no_grad()
def apply_rope(q, k, cos, sin):
    # q,k: [B,nH,d] 或 [B,nH,T,d]
    assert q.shape == k.shape
    assert q.size(-1) % 2 == 0
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    qr = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    kr = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return qr, kr
