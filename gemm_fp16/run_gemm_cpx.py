import torch
import time

# Matrix dimensions
M, K, N = 16384, 16384, 16384
num_warmup = 10
num_iter = 100
num_gpus = 2  # You can change this if needed

print(f"Using {num_gpus} GPUs")

# Preallocate per-device tensors
per_device_data = []
for device_id in range(num_gpus):
    torch.cuda.set_device(device_id)
    A = torch.randn(M, K, device=f"cuda:{device_id}", dtype=torch.float16)
    B = torch.randn(K, N, device=f"cuda:{device_id}", dtype=torch.float16)
    per_device_data.append((device_id, A, B))

# Warm-up
for device_id, A, B in per_device_data:
    for _ in range(num_warmup):
        torch.matmul(A, B)

for device_id, _, _ in per_device_data:
    torch.cuda.synchronize(device_id)

streams = []
for device_id, A, B in per_device_data:
    stream = torch.cuda.Stream(device=device_id)
    streams.append((device_id, stream, A, B))

start = time.time()
for _ in range(num_iter):
    for device_id, stream, A, B in streams:
        with torch.cuda.stream(stream):
            torch.matmul(A, B)

for device_id, stream, _, _ in streams:
    torch.cuda.synchronize(device_id)

end = time.time()

avg_time = (end - start) / num_iter

# Total FLOPs across all GPUs
# Each GPU computes 2 * M * K * N FLOPs
total_flops = 2 * M * K * N * num_gpus
tflops = total_flops / (avg_time * 1e12)

print(f"Average GEMM time over {num_iter} runs: {avg_time:.6f} seconds")
print(f"Total performance across {num_gpus} GPUs: {tflops:.2f} TFLOPS")
