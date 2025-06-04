import torch
import time

# Matrix dimensions
M, K, N = 16384, 16384, 16384
num_warmup = 10
num_iter = 10

# num_gpus = torch.cuda.device_count()
num_gpus = 2

print(f"Using {num_gpus} GPUs")

# Split rows of A and C across devices
M_per_gpu = M // num_gpus

# Store per-GPU timings
times = []

for device_id in range(num_gpus):
    torch.cuda.set_device(device_id)

    A_chunk = torch.randn(M_per_gpu, K, device=f"cuda:{device_id}", dtype=torch.float16)
    B = torch.randn(K, N, device=f"cuda:{device_id}", dtype=torch.float16)

    # Warm-up
    for _ in range(num_warmup):
        torch.matmul(A_chunk, B)
    torch.cuda.synchronize(device_id)

# Timed runs
for _ in range(num_iter):
    start_time = time.time()
    streams = []

    for device_id in range(num_gpus):
        torch.cuda.set_device(device_id)

        A_chunk = torch.randn(M_per_gpu, K, device=f"cuda:{device_id}", dtype=torch.float16)
        B = torch.randn(K, N, device=f"cuda:{device_id}", dtype=torch.float16)

        stream = torch.cuda.Stream(device_id)
        streams.append((device_id, stream, A_chunk, B))

    # Launch all matmuls on separate streams
    for device_id, stream, A_chunk, B in streams:
        with torch.cuda.stream(stream):
            torch.matmul(A_chunk, B)

    # Synchronize all devices
    for device_id, stream, _, _ in streams:
        torch.cuda.synchronize(device_id)

    end_time = time.time()
    times.append(end_time - start_time)

avg_time = sum(times) / num_iter

# Total operations across all GPUs
total_flops = 2 * M * K * N
tflops = total_flops / (avg_time * 1e12)

print(f"Average GEMM time over {num_iter} runs: {avg_time:.6f} seconds")
print(f"Total performance across {num_gpus} GPUs: {tflops:.2f} TFLOPS")
