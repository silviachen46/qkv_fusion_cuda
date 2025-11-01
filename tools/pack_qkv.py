import torch
from transformers import AutoModelForCausalLM, AutoConfig
import argparse, os

def pack_qkv(model_name, out_path):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cpu')
    # Example for one layer (you will need to iterate all layers):
    layer = model.model.layers[0] if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
    if layer is None:
        raise RuntimeError("Model structure unexpected; implement your own traversal to fetch q/k/v weights")

    # NOTE: this is model-dependent!
    q = layer.self_attn.q_proj.weight.data
    k = layer.self_attn.k_proj.weight.data
    v = layer.self_attn.v_proj.weight.data
    qb = layer.self_attn.q_proj.bias.data if layer.self_attn.q_proj.bias is not None else torch.zeros(q.size(0))
    kb = layer.self_attn.k_proj.bias.data if layer.self_attn.k_proj.bias is not None else torch.zeros(k.size(0))
    vb = layer.self_attn.v_proj.bias.data if layer.self_attn.v_proj.bias is not None else torch.zeros(v.size(0))

    # Pack [H,3H] or [3H,H] depending on your matmul layout; here we assume [H,3H] col-cat
    W_qkv = torch.cat([q.t(), k.t(), v.t()], dim=1).contiguous().t().t()  # placeholder to emphasize shape ops
    b_qkv = torch.cat([qb, kb, vb], dim=0).contiguous()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'W_qkv': W_qkv, 'b_qkv': b_qkv}, out_path)
    print(f"Saved packed weights to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF model name or local path')
    ap.add_argument('--out', required=True, help='output .pt')
    args = ap.parse_args()
    pack_qkv(args.model, args.out)
