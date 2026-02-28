// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include <torch/library.h>
#include "mix_mma_multistage.cuh"

int add(int i, int j) {
    return i + j;
}

torch::Tensor create()
{
    return torch::rand({2, 3});
}

torch::Tensor add_tensor(torch::Tensor &a, torch::Tensor &b)
{
    return a + b;
}


// Input [tokens, hidden_size] in column major. output in row major in the same
// shape.
template <int N, int BLOCK_TILE_M = 64, int BLOCK_TILE_N = 128,
          int BLOCK_SIZE = 256>
__global__ void transpose_c2r_even(const __half* input_cmajor,
                                   __half* output_rmajor, const int M) {
  using vector_t = float4;
  constexpr int vector_size = sizeof(vector_t) / sizeof(__half);  // 8
  constexpr int tile_size = BLOCK_TILE_M * BLOCK_TILE_N;
  static_assert(BLOCK_TILE_M % vector_size == 0 &&
                BLOCK_TILE_N % vector_size == 0);
  constexpr int vector_number = tile_size / vector_size;
  static_assert(vector_number % BLOCK_SIZE == 0);
  constexpr int iters = vector_number / BLOCK_SIZE;
  static_assert(BLOCK_TILE_N % iters == 0);
  static_assert(BLOCK_TILE_M % iters == 0);
  static_assert(BLOCK_TILE_M % vector_size == 0);
  static_assert(BLOCK_TILE_N % vector_size == 0);

  const int m_start_block = blockIdx.x * BLOCK_TILE_M;
  const int n_start_block = blockIdx.y * BLOCK_TILE_N;
  const __half* input_block = input_cmajor + m_start_block + n_start_block * M;
  __half* output_block = output_rmajor + m_start_block * N + n_start_block;

  __half input_frags[iters * vector_size];
  // Padding a half2 to avoid bank conflict.
  __shared__ __align__(128) __half buffer[BLOCK_TILE_M][BLOCK_TILE_N + 2];

  constexpr int n_per_iter_in = BLOCK_TILE_N / iters;
  constexpr int m_vectors = BLOCK_TILE_M / vector_size;
  int tid_m_in = threadIdx.x % m_vectors;
  int tid_n_in = threadIdx.x / m_vectors;
  int m_start_thread = tid_m_in * vector_size;
  const __half* input_thread = input_block + m_start_thread + tid_n_in * M;
  // Load global memory to registers.
  vector_t* frag_vectors = reinterpret_cast<vector_t*>(input_frags);
#pragma unroll
  for (int i = 0; i < iters; i++) {
    frag_vectors[i] = *reinterpret_cast<const vector_t*>(
        input_thread + i * (n_per_iter_in * M));
  }
  // Store data from registers to shared memory, transposing.
#pragma unroll
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < vector_size; j++) {
      // Index m: m_start_thread + j, n: tid_n + wave * n_per_iter_in
      buffer[m_start_thread + j][tid_n_in + i * n_per_iter_in] =
          input_frags[i * vector_size + j];
    }
  }
  __syncthreads();
  constexpr int m_per_iter_out = BLOCK_TILE_M / iters;
  constexpr int n_vectors = BLOCK_TILE_N / vector_size;
  int tid_m_out = threadIdx.x / n_vectors;
  int tid_n_out = threadIdx.x % n_vectors;
  // Load data from shared memory to registers.
#pragma unroll
  for (int i = 0; i < iters; i++) {
    __half2* frag_half2 =
        reinterpret_cast<__half2*>(input_frags + i * vector_size);
    __half2* buffer_half2 = reinterpret_cast<__half2*>(
        &buffer[tid_m_out + i * m_per_iter_out][tid_n_out * vector_size]);
    static_assert(vector_size % 2 == 0);
#pragma unroll
    for (int j = 0; j < vector_size / 2; j++) {
      frag_half2[j] = buffer_half2[j];
    }
  }
  // Store data from registers to global memory.
  __half* output_thread =
      output_block + tid_m_out * N + tid_n_out * vector_size;
#pragma unroll
  for (int i = 0; i < iters; i++) {
    *reinterpret_cast<vector_t*>(output_thread + i * (m_per_iter_out * N)) =
        frag_vectors[i];
  }
}

// Input [tokens, hidden_size] in column major. output in row major in the same
// shape.
template <int N, int BLOCK_TILE_M = 64, int BLOCK_TILE_N = 128,
          int BLOCK_SIZE = 256>
__global__ void transpose_c2r(const __half* input_cmajor, __half* output_rmajor,
                              const int M) {
  using vector_t = float4;
  constexpr int vector_size = sizeof(vector_t) / sizeof(__half);  // 8
  constexpr int tile_size = BLOCK_TILE_M * BLOCK_TILE_N;
  static_assert(BLOCK_TILE_M % vector_size == 0 &&
                BLOCK_TILE_N % vector_size == 0);
  constexpr int vector_number = tile_size / vector_size;
  static_assert(vector_number % BLOCK_SIZE == 0);
  constexpr int iters = vector_number / BLOCK_SIZE;
  static_assert(BLOCK_TILE_N % iters == 0);
  static_assert(BLOCK_TILE_M % iters == 0);
  static_assert(BLOCK_TILE_M % vector_size == 0);
  static_assert(BLOCK_TILE_N % vector_size == 0);

  const int m_start_block = blockIdx.x * BLOCK_TILE_M;
  const int n_start_block = blockIdx.y * BLOCK_TILE_N;
  const __half* input_block = input_cmajor + m_start_block + n_start_block * M;
  __half* output_block = output_rmajor + m_start_block * N + n_start_block;

  __half input_frags[iters * vector_size];
  // Padding a half2 to avoid bank conflict.
  __shared__ __align__(128) __half buffer[BLOCK_TILE_M][BLOCK_TILE_N + 2];

  int m_left_block = M - m_start_block;
  if (m_left_block >= BLOCK_TILE_M) {
    constexpr int n_per_iter_in = BLOCK_TILE_N / iters;
    constexpr int m_vectors = BLOCK_TILE_M / vector_size;
    int tid_m_in = threadIdx.x % m_vectors;
    int tid_n_in = threadIdx.x / m_vectors;
    int m_start_thread = tid_m_in * vector_size;
    const __half* input_thread = input_block + m_start_thread + tid_n_in * M;
    // Load global memory to registers.
    vector_t* frag_vectors = reinterpret_cast<vector_t*>(input_frags);
#pragma unroll
    for (int i = 0; i < iters; i++) {
      __half* frag = reinterpret_cast<__half*>(input_frags + i * vector_size);
      // The memory address may not be aligned, so we can't use vector load.
#pragma unroll
      for (int j = 0; j < vector_size; j++) {
        frag[j] = input_thread[j + i * (n_per_iter_in * M)];
      }
    }
    // Store data from registers to shared memory, transposing.
#pragma unroll
    for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int j = 0; j < vector_size; j++) {
        // Index m: m_start_thread + j, n: tid_n + wave * n_per_iter_in
        buffer[m_start_thread + j][tid_n_in + i * n_per_iter_in] =
            input_frags[i * vector_size + j];
      }
    }
    __syncthreads();
    constexpr int m_per_iter_out = BLOCK_TILE_M / iters;
    constexpr int n_vectors = BLOCK_TILE_N / vector_size;
    int tid_m_out = threadIdx.x / n_vectors;
    int tid_n_out = threadIdx.x % n_vectors;
    // Load data from shared memory to registers.
#pragma unroll
    for (int i = 0; i < iters; i++) {
      __half2* frag_half2 =
          reinterpret_cast<__half2*>(input_frags + i * vector_size);
      __half2* buffer_half2 = reinterpret_cast<__half2*>(
          &buffer[tid_m_out + i * m_per_iter_out][tid_n_out * vector_size]);
      static_assert(vector_size % 2 == 0);
#pragma unroll
      for (int j = 0; j < vector_size / 2; j++) {
        frag_half2[j] = buffer_half2[j];
      }
    }
    // Store data from registers to global memory.
    __half* output_thread =
        output_block + tid_m_out * N + tid_n_out * vector_size;
#pragma unroll
    for (int i = 0; i < iters; i++) {
      *reinterpret_cast<vector_t*>(output_thread + i * (m_per_iter_out * N)) =
          frag_vectors[i];
    }
  } else if (m_left_block > 0) {
    // The tail block.
    constexpr int n_per_iter_in = BLOCK_TILE_N / iters;
    constexpr int m_vectors = BLOCK_TILE_M / vector_size;
    int tid_m_in = threadIdx.x % m_vectors;
    int tid_n_in = threadIdx.x / m_vectors;
    int m_start_thread = tid_m_in * vector_size;
    const __half* input_thread = input_block + m_start_thread + tid_n_in * M;
    int m_left_thread = m_left_block - m_start_thread;
    // Load global memory to registers.
    vector_t* frag_vectors = reinterpret_cast<vector_t*>(input_frags);
    if (m_left_thread >= vector_size) {
#pragma unroll
      for (int i = 0; i < iters; i++) {
        __half* frag = reinterpret_cast<__half*>(input_frags + i * vector_size);
        // The memory address may not be aligned, so we can't use vector load.
#pragma unroll
        for (int j = 0; j < vector_size; j++) {
          frag[j] = input_thread[j + i * (n_per_iter_in * M)];
        }
      }
      // Store data from registers to shared memory, transposing.
#pragma unroll
      for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < vector_size; j++) {
          // Index m: m_start_block + j, n: tid_n + wave * n_per_iter_in
          buffer[m_start_thread + j][tid_n_in + i * n_per_iter_in] =
              input_frags[i * vector_size + j];
        }
      }
    } else if (m_left_thread > 0) {
      // The tail of the tail block. Can't use vector load.
      // Load global memory to registers.
#pragma unroll
      for (int i = 0; i < iters; i++) {
        const __half* input_tail_ptr = input_thread + i * (n_per_iter_in * M);
        for (int j = 0; j < m_left_thread; j++) {
          input_frags[i * vector_size + j] = input_tail_ptr[j];
        }
      }
      // Store data from registers to shared memory, transposing.
#pragma unroll
      for (int i = 0; i < iters; i++) {
        for (int j = 0; j < m_left_thread; j++) {
          buffer[m_start_thread + j][tid_n_in + i * n_per_iter_in] =
              input_frags[i * vector_size + j];
        }
      }
    }
    __syncthreads();

    if (m_left_block <= 0) {
      return;
    }

    constexpr int m_per_iter_out = BLOCK_TILE_M / iters;
    constexpr int n_vectors = BLOCK_TILE_N / vector_size;
    int tid_m_out = threadIdx.x / n_vectors;
    int tid_n_out = threadIdx.x % n_vectors;
    // Load data from shared memory to registers.
#pragma unroll
    for (int i = 0; i < iters; i++) {
      int m_idx_in_block = tid_m_out + i * m_per_iter_out;
      if (m_idx_in_block >= m_left_block) {
        break;
      }
      __half2* frag_half2 =
          reinterpret_cast<__half2*>(input_frags + i * vector_size);
      __half2* buffer_half2 = reinterpret_cast<__half2*>(
          &buffer[m_idx_in_block][tid_n_out * vector_size]);
      static_assert(vector_size % 2 == 0);
#pragma unroll
      for (int j = 0; j < vector_size / 2; j++) {
        frag_half2[j] = buffer_half2[j];
      }
    }
    // Store data from registers to global memory.
    __half* output_thread_base = output_block + tid_n_out * vector_size;
#pragma unroll
    for (int i = 0; i < iters; i++) {
      int m_idx_in_block = tid_m_out + i * m_per_iter_out;
      if (m_idx_in_block >= m_left_block) {
        break;
      }
      *reinterpret_cast<vector_t*>(output_thread_base + m_idx_in_block * N) =
          frag_vectors[i];
    }
  }
}

torch::Tensor transpose(torch::Tensor &input) {
  // TODO: does not performan well when M is smaller than 8.
  constexpr int BLOCK_TILE_M = 64;
  constexpr int BLOCK_TILE_N = 64;
  constexpr int BLOCK_SIZE = 256;
  int N = input.size(0);
  int M = input.size(1);
  assert(N % BLOCK_TILE_N == 0);

  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
  torch::Tensor output = torch::empty({M, N}, options);

  __half *input_ptr = reinterpret_cast<__half*>(input.data_ptr());
  __half *output_ptr = reinterpret_cast<__half*>(output.data_ptr());

  dim3 grid((M + BLOCK_TILE_M - 1) / BLOCK_TILE_M, N / BLOCK_TILE_N);
  bool m_even = M % BLOCK_TILE_M == 0;
  if (m_even) {
    if (N == 4096) {
      transpose_c2r_even<4096, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 14336) {
      transpose_c2r_even<14336, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 1024) {
      transpose_c2r_even<1024, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 6144) {
      transpose_c2r_even<6144, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 28672) {
      transpose_c2r_even<28672, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else {
      printf("Unsupported N: %d\n", N);
    }
  } else {
    if (N == 4096) {
      transpose_c2r<4096, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 14336) {
      transpose_c2r<14336, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 1024) {
      transpose_c2r<1024, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 6144) {
      transpose_c2r<6144, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else if (N == 28672) {
      transpose_c2r<28672, BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_SIZE>
          <<<grid, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(input_ptr, output_ptr, M);
    } else {
      printf("Unsupported N: %d\n", N);
    }
  }
  return output;
}


template <typename T>
__host__ __device__ __forceinline__ T max(const T lhs, const T rhs);

template <>
__host__ __device__ __forceinline__ __half max(const __half lhs, const __half rhs) {
#if true
  // Intrinsic limited to Ampere + newer
  return __hmax(lhs, rhs);
#else
  return (lhs > rhs) ? lhs : rhs;
#endif
}

template <>
__host__ __device__ __forceinline__ __half2 max(const __half2 lhs, const __half2 rhs) {
#if true
  return __hmax2(lhs, rhs);
#else
  __half2 ret_val;
  ret_val.x = (lhs.x > rhs.x) ? lhs.x : rhs.x;
  ret_val.y = (lhs.y > rhs.y) ? lhs.y : rhs.y;
  return ret_val;
#endif
}

__device__ inline int8_t half2s8(half val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  union {
    half fp16;
    int16_t int16_in;
  };

  fp16 = val;
  asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
  return int8[0];
}


// TODO: support bf16.
template <int GROUP_SIZE = 128, int ROW_TILE = 1, int BLOCK_SIZE = 128>
__global__ void quantize_fg_sym_f16s8(int8_t* quantized_rmajor,
                                      __half* scales_cmajor,
                                      const __half* input_rmajor,
                                      const int tokens, const int hidden_size,
                                      const int tokens_round_even  // for scale
) {
  static_assert(GROUP_SIZE == 128, "GROUP_SIZE must be 128");

  // A group is processed by a warp. A warp will process a tile of
  // `ROW_TILE * GROUP_SIZE` elements.
  constexpr int kWarpSize = 32;
  constexpr int elems_per_block = BLOCK_SIZE / kWarpSize * GROUP_SIZE;
  // Every thread block will process kWarpSize groups, and will 'pad' (skip) the
  // tail.
  const int warps_per_row = (hidden_size + elems_per_block - 1) /
                            elems_per_block * (BLOCK_SIZE / kWarpSize);
  constexpr int elems_per_thread_per_tile = GROUP_SIZE / kWarpSize;  // 4
  constexpr int elems_per_thread = ROW_TILE * elems_per_thread_per_tile;
  static_assert(elems_per_thread < 256);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane_id = tid % kWarpSize;
  int warp_row = warp_id / warps_per_row;
  int warp_col = warp_id % warps_per_row;
  int row_offset = warp_row * ROW_TILE;
  int col_offset = warp_col * GROUP_SIZE;
  const __half* input_ptr =
      input_rmajor + row_offset * hidden_size + col_offset;
  int8_t* quantized_ptr =
      quantized_rmajor + row_offset * hidden_size + col_offset;

  if (row_offset >= tokens || col_offset >= hidden_size) {
    return;
  }

  // Load input data into register.
  __half input_frag[ROW_TILE][elems_per_thread_per_tile];
  using vector_input_t = float2;  // 128 * 2 / 32 = 8 bytes per thread.
  static_assert(sizeof(vector_input_t) * kWarpSize ==
                sizeof(__half) * GROUP_SIZE);
  using vector_output_t = float;  // 128 * 1 / 32 = 4 bytes per thread.
#pragma unroll
  for (int i = 0; i < ROW_TILE; ++i) {
    vector_input_t* input_frag_vector =
        reinterpret_cast<vector_input_t*>(input_frag[i]);
    const vector_input_t* input_vector_thread =
        reinterpret_cast<const vector_input_t*>(input_ptr + i * hidden_size);
    *input_frag_vector = input_vector_thread[lane_id];
  }

  // Per group max.
  // constexpr __half2 max_init = {0.0f, 0.0f};
  __half max_val[ROW_TILE];
#pragma unroll
  for (int i = 0; i < ROW_TILE; ++i) {
    __half2* half2_frag = reinterpret_cast<__half2*>(input_frag[i]);
    // Given group_size 128, there are only 2 half2 elements per thread.
    __half2 max_val_2 = max(__habs2(half2_frag[0]), __habs2(half2_frag[1]));
    max_val[i] = max(max_val_2.x, max_val_2.y);
  }
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset *= 2) {
#pragma unroll
    for (int i = 0; i < ROW_TILE; ++i) {
      max_val[i] = max(max_val[i], __shfl_xor_sync(0xffffffff, max_val[i],
                                                   offset, kWarpSize));
    }
  }

  // Quantize and store.
  __half max_int = 127.0f;  // 2 << (8 - 1) - 1
#pragma unroll
  for (int i = 0; i < ROW_TILE; ++i) {
    __half rscale;
    if (lane_id == 0) {
      __half scale = max_val[i] / max_int;
      // Write scales in column major. May vectorize with shared memory.
      scales_cmajor[warp_col * tokens_round_even + row_offset + i] = scale;
      rscale = __hdiv(1.0, scale);
    }
    rscale = __shfl_sync(0xffffffff, rscale, 0, kWarpSize);
    vector_output_t* output_vector_thread =
        reinterpret_cast<vector_output_t*>(quantized_ptr + i * hidden_size);
    int8_t res[elems_per_thread_per_tile];
    __half2* half2_frag = reinterpret_cast<__half2*>(input_frag[i]);
#pragma unroll
    for (int j = 0; j < elems_per_thread_per_tile / 2; ++j) {
      __half2 elem = half2_frag[j];
      elem = __hmul2(elem, {rscale, rscale});
      res[j * 2] = half2s8(elem.x);
      res[j * 2 + 1] = half2s8(elem.y);
    }
    output_vector_thread[lane_id] = *reinterpret_cast<vector_output_t*>(res);
  }
}


std::tuple<torch::Tensor, torch::Tensor> quantize(const torch::Tensor &input_d) {
  TORCH_CHECK(input_d.dim() == 2, "Input tensor must be 2D");
  TORCH_CHECK(input_d.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(input_d.dtype() == torch::kFloat16, "Input tensor must be of type Float16");
  const int M = input_d.size(0);
  const int N = input_d.size(1);
  constexpr int group_size = 128;
  TORCH_CHECK(N % group_size == 0, "N must be divisible by group_size");

  const int M_round_even = M + (M % 2);
  const auto options_scales_d = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
  torch::Tensor scales_d = torch::empty({N/group_size, M_round_even}, options_scales_d);
  const auto options_quantized = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt8);
  torch::Tensor quantized_d = torch::empty({M, N}, options_quantized);

  // It assumes row-major input, MxN.
  constexpr int block_size = 128;
  constexpr int kWarpSize = 32;
  constexpr int elems_per_block_ctile = block_size / kWarpSize * group_size;
  const int blocks_n = (N + elems_per_block_ctile - 1) / elems_per_block_ctile;

  __half *input_d_ptr = reinterpret_cast<__half*>(input_d.data_ptr());
  int8_t *quantized_d_ptr = reinterpret_cast<int8_t*>(quantized_d.data_ptr());
  __half *scales_d_ptr = reinterpret_cast<__half*>(scales_d.data_ptr());

  constexpr int row_tile = 1;
  const int blocks_m = M / row_tile;
  const int blocks = blocks_m * blocks_n;
  quantize_fg_sym_f16s8<group_size, row_tile, block_size>
      <<<blocks, block_size, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        quantized_d_ptr, scales_d_ptr, input_d_ptr, M, N, M_round_even);
  
  return std::make_tuple(quantized_d, scales_d);
}

std::unique_ptr<mixllm::LinearMixLLM> mixllm::LinearMixLLM::instance;
std::mutex mixllm::LinearMixLLM::instance_mutex;

torch::Tensor gemm(
          torch::Tensor &matrix_A,
          torch::Tensor &matrix_scale_act, 
          torch::Tensor &matrix_zero,
          torch::Tensor &matrix_scale_int8,
          torch::Tensor &matrix_scale_int4,
          torch::Tensor &matrix_indices_int8,
          torch::Tensor &matrix_indices_int4,
          torch::Tensor &matrix_B_int8,
          torch::Tensor &matrix_B_interleaved) {
  mixllm::LinearMixLLM::initialize();
  const int M = matrix_A.size(0);
  const int N = matrix_indices_int8.numel() + matrix_indices_int4.numel();
  const bool is_row_major 
    = (matrix_indices_int8.numel() == 0 || matrix_indices_int4.numel() == 0);
  const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
  torch::Tensor matrix_C_computed = is_row_major ?
      torch::empty({M, N}, options) :
      torch::empty({N, M}, options);

  mixllm::LinearMixLLM::run(
      matrix_C_computed, matrix_A, matrix_scale_act, 
      matrix_zero, matrix_scale_int8, matrix_scale_int4, 
      matrix_indices_int8, matrix_indices_int4, 
      matrix_B_int8, matrix_B_interleaved);

  return matrix_C_computed;
}

// Defines the operators
TORCH_LIBRARY(kernels_mixllm, m) {
  m.def("quantize(Tensor a) -> (Tensor, Tensor)");
  m.impl("quantize", torch::kCUDA, &quantize);

  m.def("transpose(Tensor a) -> Tensor");
  m.impl("transpose", torch::kCUDA, &transpose);

  m.def("gemm(Tensor matrix_A, "
         "Tensor matrix_scale_act, Tensor matrix_zero, "
         "Tensor matrix_scale_int8, Tensor matrix_scale_int4, "
         "Tensor matrix_indices_int8, Tensor matrix_indices_int4, "
         "Tensor matrix_B_int8, Tensor matrix_B_interleaved) -> Tensor");
  m.impl("gemm", torch::kCUDA, &gemm);
}
