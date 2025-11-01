import torch, pytest
def test_shapes():
    B,H = 2, 8
    x = torch.randn(B,H, device='cpu')
    W = torch.randn(H, 3*H, device='cpu')
    b = torch.randn(3*H, device='cpu')
    y = x @ W + b
    assert y.shape == (B, 3*H)
