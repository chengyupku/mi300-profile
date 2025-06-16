import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Matrix dimensions
M, K, N = 16384, 16384, 16384

# Number of runs to average
num_warmup = 10
num_iter = 100

# Generate input matrices
A = torch.randn(M, K, device=device, dtype=torch.float16)
B = torch.randn(K, N, device=device, dtype=torch.float16)

# Warm-up
for _ in range(num_warmup):
    torch.matmul(A, B)

torch.cuda.synchronize()
start = time.time()

for _ in range(num_iter):
    C = torch.matmul(A, B)

torch.cuda.synchronize()
end = time.time()
avg_time = (end - start) / num_iter

# Compute TFLOPS: (2 * M * K * N) / (time * 1e12)
tflops = (2 * M * K * N) / (avg_time * 1e12)

print(f"Average GEMM time over {num_iter} runs: {avg_time:.6f} seconds")
print(f"Performance: {tflops:.2f} TFLOPS")
