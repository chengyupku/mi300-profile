import torch
import time

from mla_decode_rope import (
    _decode_grouped_att_m_fwd_rope,
    decode_attention_fwd_grouped_rope,
)

from rotary_embedding import DeepseekScalingRotaryEmbedding

def ref_preprocess(kv_cache, kv_lora_rank):
    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    return k_input, v_input


def input_helper(
    B,
    H,
    S,
    kv_lora_rank,
    rotary_dim,
    qk_rope_head_dim,
    num_kv_splits,
    dtype,
    device,
    rope_base=10,
    rope_max_seq_len=16324,
    rope_scaling=1.0,
    is_neox_style=True,
):
    q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(
        B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )

    # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
    kv_indptr = torch.arange(B + 1, device=device) * S
    kv_indices = torch.arange(B * S, device=device)

    attn_logits = torch.empty(
        B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device
    )

    rotary_emb = DeepseekScalingRotaryEmbedding(
        qk_rope_head_dim,
        rotary_dim,
        rope_max_seq_len,
        rope_base,
        is_neox_style,
        rope_scaling,
        q.dtype,
        device=device,
    )

    positions = (
        torch.tensor([S], device=device).unsqueeze(0).repeat(B, 1)
    )  # k positions and q position as last

    return kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions

def test_op_fwd_rope_integration(
    B,
    H,
    S,
    kv_lora_rank,
    qk_rope_head_dim,
    rotary_dim,
    dtype,
    use_rope,
    is_neox_style,
    num_kv_splits=2,
    sm_scale=1.0,
    logit_cap=0.0,
    device="cuda",
):
    warmup = 10
    iters = 10

    torch.manual_seed(0)

    kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions = (
        input_helper(
            B,
            H,
            S,
            kv_lora_rank,
            rotary_dim,
            qk_rope_head_dim,
            num_kv_splits,
            dtype,
            device,
            is_neox_style=is_neox_style,
        )
    )

    # we need to return the rope'd k_pe_tokens to be saved in cache
    k_pe_tokens = torch.empty(B, qk_rope_head_dim, dtype=kv_cache.dtype, device=device)
    tri_o = torch.empty(B, H, kv_lora_rank, dtype=kv_cache.dtype, device=device)

    k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)


    for i in range(warmup):
        _ = decode_attention_fwd_grouped_rope(
                q,
                k_input,
                v_input,
                tri_o,
                kv_indptr,
                kv_indices,
                k_pe_tokens if use_rope else None,
                kv_lora_rank,
                rotary_dim if use_rope else None,
                rotary_emb.cos_sin_cache if use_rope else None,
                positions if use_rope else None,
                attn_logits,
                num_kv_splits,
                sm_scale,
                logit_cap,
                use_rope,
                is_neox_style,
            )

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iters):
        _ = decode_attention_fwd_grouped_rope(
                q,
                k_input,
                v_input,
                tri_o,
                kv_indptr,
                kv_indices,
                k_pe_tokens if use_rope else None,
                kv_lora_rank,
                rotary_dim if use_rope else None,
                rotary_emb.cos_sin_cache if use_rope else None,
                positions if use_rope else None,
                attn_logits,
                num_kv_splits,
                sm_scale,
                logit_cap,
                use_rope,
                is_neox_style,
            )
        
    torch.cuda.synchronize()
    end_time = time.time()
    total_ms = (end_time - start_time) * 1e3 / iters

    qk_flops = 2 * B * H * S * (qk_rope_head_dim + qk_rope_head_dim)
    pv_flops = 2 * B * H * S * qk_rope_head_dim
    total_flops = qk_flops + pv_flops


    print(f"Time taken: {total_ms:.2f} ms")
    print(f"TFLOPS: {total_flops / (total_ms * 1e9):.2f}")

    # decode_attention_fwd_grouped_rope(
    #     q,
    #     k_input,
    #     v_input,
    #     tri_o,
    #     kv_indptr,
    #     kv_indices,
    #     k_pe_tokens if use_rope else None,
    #     kv_lora_rank,
    #     rotary_dim if use_rope else None,
    #     rotary_emb.cos_sin_cache if use_rope else None,
    #     positions if use_rope else None,
    #     attn_logits,
    #     num_kv_splits,
    #     sm_scale,
    #     logit_cap,
    #     use_rope,
    #     is_neox_style,
    # )


if __name__ == "__main__":
    test_op_fwd_rope_integration(
        B=128,
        H=128,
        S=8192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        rotary_dim=64,
        dtype=torch.float16,
        use_rope=True,
        is_neox_style=True,
        num_kv_splits=2,
        sm_scale=1.0,
        logit_cap=0.0,
        device="cuda",
    )