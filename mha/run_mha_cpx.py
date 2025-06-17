import torch
import time
from torch.nn.functional import scaled_dot_product_attention

BATCH = 16
HEADS = 64
N_CTX_Q = 8192
N_CTX_K = 8192
D_HEAD = 128

query = torch.randn(BATCH, HEADS, N_CTX_Q, D_HEAD, device="cuda")
key = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device="cuda")
value = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device="cuda")

num_warmup = 10
num_iter = 100
num_gpus = 1  # You can change this if needed

print(f"Using {num_gpus} GPUs")

# Preallocate per-device tensors
per_device_data = []
for device_id in range(num_gpus):
    torch.cuda.set_device(device_id)
    query = torch.randn(BATCH, HEADS, N_CTX_Q, D_HEAD, device=f"cuda:{device_id}", dtype=torch.float16)
    key = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device=f"cuda:{device_id}", dtype=torch.float16)
    value = torch.randn(BATCH, HEADS, N_CTX_K, D_HEAD, device=f"cuda:{device_id}", dtype=torch.float16)
    per_device_data.append((device_id, query, key, value))

# Warm-up
for device_id, query, key, value in per_device_data:
    for _ in range(num_warmup):
        _ = scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

for device_id, _, _, _ in per_device_data:
    torch.cuda.synchronize(device_id)

streams = []
for device_id, query, key, value in per_device_data:
    stream = torch.cuda.Stream(device=device_id)
    streams.append((device_id, stream, query, key, value))

start = time.time()
for _ in range(num_iter):
    for device_id, stream, query, key, value in streams:
        with torch.cuda.stream(stream):
            _ = scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

for device_id, stream, _, _, _ in streams:
    torch.cuda.synchronize(device_id)

end = time.time()

avg_time = (end - start) / num_iter

# Total FLOPs across all GPUs
# Each GPU computes 2 * M * K * N FLOPs
total_flops = 4 * BATCH * HEADS * N_CTX_Q * N_CTX_K * D_HEAD * num_gpus
tflops = total_flops / (avg_time * 1e12)

print(f"Average GEMM time over {num_iter} runs: {avg_time:.6f} seconds")
print(f"Total performance across {num_gpus} GPUs: {tflops:.2f} TFLOPS")