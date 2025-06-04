
// const int M = 16384;
// const int N = 16384;
const int M = 8192;
const int N = 8192;
const int K = 8192;

#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <rocwmma/rocwmma.hpp>

// #include <hip/hcc_detail/hip_fp16_math_fwd.h>

// #define hpow __ocml_pown_f16
// #define hsqrt __ocml_sqrt_f16

// #define htanh(x) __float2half_rn(tanh(__half2float(x)))
// #define htan(x) __float2half_rn(tanf(__half2float(x)))
// #define hatan(x) __float2half_rn(atanf(__half2float(x)))
// #define herf(x) __float2half_rn(erff(__half2float(x)))
// #define hexp(x) __float2half_rn(expf(__half2float(x)))

// #define HIPRT_INF_F        __int_as_float(0x7f800000)
// #define HIPRT_NAN_F        __int_as_float(0x7fffffff)
// #define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
// #define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
// #define HIPRT_NEG_ZERO_F   __int_as_float(0x80000000)
// #define HIPRT_ZERO_F       0.0f
// #define HIPRT_ONE_F        1.0f

// /* double precision constants */
// #define HIPRT_INF          __hiloint2double(0x7ff00000, 0x00000000)
// #define HIPRT_NAN          __hiloint2double(0xfff80000, 0x00000000)

// #define max(a, b) (((a) > (b)) ? (a) : (b))
// #define min(a, b) (((a) < (b)) ? (a) : (b))

using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#define half _Float16
#define __float2half_rn(x) half(x)

// #define htanh tanhf
// #define htan tanf
// #define hatan atanf
// #define herf erff
// #include <hip/hcc_detail/hip_fp16_math_fwd.h>
// #define hpow __ocml_pown_f16
// #define hsqrt __ocml_sqrt_f16
// #define hexp __ocml_exp_f16

// // Pack two half values.
// inline __device__ __host__ unsigned
// __pack_half2(const half x, const half y) {
//   unsigned v0 = *((unsigned short *)&x);
//   unsigned v1 = *((unsigned short *)&y);
//   return (v1 << 16) | v0;
// }

// // There is no make_int8 in cuda, but TVM codegen seem to use it
// inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4,
// int x5, int x6, int x7) {
//   int2 i0 = make_int2(x0, x1);
//   int2 i1 = make_int2(x2, x3);
//   int2 i2 = make_int2(x4, x5);
//   int2 i3 = make_int2(x6, x7);
//   long long l0 = *(long long*)&i0;
//   long long l1 = *(long long*)&i1;
//   long long l2 = *(long long*)&i2;
//   long long l3 = *(long long*)&i3;
//   return make_longlong4(l0, l1, l2, l3);
// }

using float16_t = _Float16;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 =
    __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 =
    __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using bfloat16_t = hip_bfloat16;
using bfloat16x4 =
    __attribute__((__vector_size__(4 * sizeof(bfloat16_t)))) float16_t;

__global__ void __launch_bounds__(256)
    pure_mfma_kernel(half *__restrict__ A, half *__restrict__ B,
                     float *__restrict__ C) {

  float C_warp[64];
  __shared__ half A_shared[4096];
  __shared__ half B_shared[4096];
  half A_shared_warp[16];
  half B_shared_warp[16];

  const int MAX_BLOCK_N = 10;
  const auto baseBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;
  const auto totalPanel =
      (gridDim.x * gridDim.y + MAX_BLOCK_N * gridDim.x - 1) /
      (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N * gridDim.x);
  const auto strideLd =
      panelIdx + 1 < totalPanel
          ? MAX_BLOCK_N
          : (totalBlock - panelIdx * (MAX_BLOCK_N * gridDim.x)) / gridDim.x;
  const auto bx =
      (panelIdx & 1)
          ? gridDim.x -
                (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) / strideLd -
                1
          : (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) / strideLd;
  const auto by =
      (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) % strideLd +
      panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);

  for (int i_2_init = 0; i_2_init < 4; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        C_warp[(((i_2_init * 16) + (j_2_init * 4)) + local_id)] = 0.000000e+00f;
      }
    }
  }
  // for (int k_0 = 0; k_0 < 1024; ++k_0) {
  for (int k_0 = 0; k_0 < int(K / 16); ++k_0) {
    __syncthreads();
    // #pragma unroll
    // for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 2;
    // ++ax0_ax1_ax2_ax3_0_fused_0) {
    //   *(uint4*)(A_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 2048) +
    //   (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) +
    //   (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) *
    //   1048576) + (ax0_ax1_ax2_ax3_0_fused_0 * 524288)) + (((int)threadIdx.y)
    //   * 262144)) + (((int)threadIdx.z) * 131072)) + (k_0 * 512)) +
    //   (((int)threadIdx.x) * 8)));
    // }
    // #pragma unroll
    // for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 <
    // 2; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
    //   *(uint4*)(B_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) +
    //   (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) +
    //   (((int)threadIdx.x) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.x) *
    //   1048576) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 524288)) +
    //   (((int)threadIdx.y) * 262144)) + (((int)threadIdx.z) * 131072)) + (k_0
    //   * 512)) + (((int)threadIdx.x) * 8)));
    // }
    // __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
          // A_shared_warp[((ax0 * 4) + local_id_1)] =
          // A_shared[(((((((int)threadIdx.y) * 2048) + (ax0 * 512)) + (k_1 *
          // 256)) + (((int)threadIdx.x) * 4)) + local_id_1)];
          A_shared_warp[((ax0 * 4) + local_id_1)] = half{1.0};
        }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        for (int local_id_2 = 0; local_id_2 < 4; ++local_id_2) {
          // B_shared_warp[((ax0_1 * 4) + local_id_2)] =
          // B_shared[(((((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (k_1 *
          // 256)) + (((int)threadIdx.x) * 4)) + local_id_2)];
          B_shared_warp[((ax0_1 * 4) + local_id_2)] = half{1.0};
        }
      }
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          {
            *(((float32x4 *)C_warp) + ((i_2 * 4) + j_2)) =
                __builtin_amdgcn_mfma_f32_16x16x16f16(
                    *(((float16x4 *)A_shared_warp) + i_2),
                    *(((float16x4 *)B_shared_warp) + j_2),
                    *(((float32x4 *)C_warp) + ((i_2 * 4) + j_2)), 0, 0, 0);
          };
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        (&(C[(
            (((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 524288)) +
               (ax0_2 * 131072)) +
              (((int)blockIdx.x) * 2048)) +
             (((int)threadIdx.z) * 1024)) +
            (ax1 * 256))]))[(((((threadIdx.x / 16) * 4) + local_id) * 16) +
                             (threadIdx.x % 16))] =
            (float)C_warp[((ax0_2 * 16) + (ax1 * 4)) + local_id];
      };
    }
  }
}

int main() {
  hipSetDevice(0);

  half *h_A = new half[M * K];
  half *h_B = new half[K * N];
  float *h_C = new float[M * N];

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<half>(static_cast<float>(rand()) /
                               static_cast<float>(RAND_MAX));
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<half>(static_cast<float>(rand()) /
                               static_cast<float>(RAND_MAX));
  }

  half *d_A;
  half *d_B;
  float *d_C;

  hipMalloc(&d_A, M * K * sizeof(half));
  hipMalloc(&d_B, K * N * sizeof(half));
  hipMalloc(&d_C, M * N * sizeof(float));

  hipMemcpy(d_A, h_A, M * K * sizeof(half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, K * N * sizeof(half), hipMemcpyHostToDevice);

  dim3 blockDim(256, 512, 1);
  dim3 gridDim(64, 1, 2);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, 0);

  // hipLaunchKernelGGL(Fused, gridDim, blockDim, 0, 0, d_A, d_B, d_C);

  // pure_mfma_kernel<<<dim3(64, 64, 1), dim3(64, 2, 2)>>>(d_A, d_B, d_C);
  // 8192/64=
  // pure_mfma_kernel<<<dim3(int(M/64), int(N/64), 1), dim3(64, 2, 2)>>>(d_A,
  // d_B, d_C);
  const int grid_M = M / 128;
  const int grid_N = N / 128;
  // pure_mfma_kernel<<<dim3(128, 128, 1), dim3(64, 2, 2)>>>(d_A, d_B, d_C);
  pure_mfma_kernel<<<dim3(grid_M, grid_N, 1), dim3(64, 2, 2)>>>(d_A, d_B, d_C);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

  std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

  std::cout << "TFLOPs = " << (2.0 * M * N * K * 2 / milliseconds / 1e9)
            << std::endl;

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}