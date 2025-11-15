import torch

@torch.no_grad()
def apply_rope(q, k, cos, sin):
    # q,k: [B, nH, d], cos/sin: [1, 1, d] 或 [B, 1, d] 或 [1, 1, d]
    d = q.size(-1)
    assert d % 2 == 0, "head_dim must be even for RoPE"
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    qr = torch.stack([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1).flatten(-2)
    kr = torch.stack([k1*cos - k2*sin, k1*sin + k2*cos], dim=-1).flatten(-2)
    return qr, kr

@torch.no_grad()
def precompute_rope(head_dim: int, pos: int, device, dtype=torch.float32):
    # 只做单步 decode：给定 token_pos 取出这一位的 cos/sin
    # θ = 10000^{-2i/d}, i 为维度偶/奇索引
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.tensor([pos], device=device, dtype=dtype)
    freqs = torch.einsum("p,d->pd", t, inv_freq)       # [1, half]
    cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)[None, None, :]  # [1,1,d]
    sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)[None, None, :]
    return cos.to(dtype), sin.to(dtype)
