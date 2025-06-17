import torch
import time
from torch.nn.functional import scaled_dot_product_attention

BATCH = 16
HEADS = 64
N_CTX_Q = 8192
N_CTX_K = 8192
D_HEAD = 128

warmup = 10
iters = 10

query = torch.randn(BATCH, HEADS, N_CTX_Q, D_HEAD, device="cuda", dtype=torch.float16)
key = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device="cuda", dtype=torch.float16)
value = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device="cuda", dtype=torch.float16)

for i in range(warmup):
    _ = scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

torch.cuda.synchronize()
start_time = time.time()
for i in range(iters):
    _ = scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
torch.cuda.synchronize()
end_time = time.time()
total_ms = (end_time - start_time) * 1e3 / iters

total_flops = (4 * BATCH * HEADS * N_CTX_Q * N_CTX_K * D_HEAD)


print(f"Time taken: {total_ms:.2f} ms")
print(f"TFLOPS: {total_flops / (total_ms * 1e9):.2f}")