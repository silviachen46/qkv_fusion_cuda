# tests/test_python_fusion_vs_baseline.py
import torch
from transformers import AutoModelForCausalLM
from torch_integration.fused_qkv_python import fused_qkv_python

@torch.no_grad()
def test_qkv_equivalence():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 选可公开下载的小模型
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    ).to(device).eval()

    # 取第0层（LLaMA结构）
    layer = model.model.layers[0]

    # hidden size / head 数从 config 读取更稳妥
    H = layer.self_attn.q_proj.weight.shape[1]              # in_features
    n_heads = getattr(model.config, "num_attention_heads")  # e.g. 2/4/...
    head_dim = H // n_heads

    x = torch.randn(2, H, device=device)

    def proj(x, w, b):
        y = x @ w.T
        if b is not None:
            y = y + b
        return y

    # baseline：三次 matmul
    q_ref = proj(x, layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias)
    k_ref = proj(x, layer.self_attn.k_proj.weight, layer.self_attn.k_proj.bias)
    v_ref = proj(x, layer.self_attn.v_proj.weight, layer.self_attn.v_proj.bias)

    # 打包一次投影
    W_qkv = torch.cat(
        [
            layer.self_attn.q_proj.weight.T,
            layer.self_attn.k_proj.weight.T,
            layer.self_attn.v_proj.weight.T,
        ],
        dim=1,
    ).to(device)

    def zero_if_none(bias, H):
        return bias if bias is not None else torch.zeros(H, device=device)

    b_qkv = torch.cat(
        [
            zero_if_none(layer.self_attn.q_proj.bias, H),
            zero_if_none(layer.self_attn.k_proj.bias, H),
            zero_if_none(layer.self_attn.v_proj.bias, H),
        ],
        dim=0,
    ).to(device)

    q, k, v = fused_qkv_python(x, W_qkv, b_qkv, n_heads)

    q_ref = q_ref.view(2, n_heads, head_dim)
    k_ref = k_ref.view(2, n_heads, head_dim)
    v_ref = v_ref.view(2, n_heads, head_dim)

    torch.testing.assert_close(q, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v, v_ref, rtol=1e-5, atol=1e-5)
