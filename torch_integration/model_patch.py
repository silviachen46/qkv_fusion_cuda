# Placeholder: add logic to replace HF model's q_proj/k_proj/v_proj with FusedQKVLinear
# You will:
# - load packed weights (W_qkv, b_qkv)
# - create FusedQKVLinear and assign into transformer blocks
# - ensure forward() passes token_pos & batch_slots
