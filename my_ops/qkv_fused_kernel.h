#pragma once
#include <torch/extension.h>

torch::Tensor qkv_fused_forward(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor b_qkv,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    torch::Tensor kv_cache_k,
    torch::Tensor kv_cache_v,
    int64_t layer_id,
    int64_t token_pos,
    double rope_scale,
    torch::Tensor batch_slots);
