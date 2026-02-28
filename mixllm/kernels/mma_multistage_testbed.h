/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Modifications Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

/*! \file
    \brief Unit testbed for kernel-level GEMM
*/

#pragma once

#include "mq_numeric_conversion.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/command_line.h"
#include <fstream>
#include <torch/extension.h>
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "mq_mma_multistage.h"
#include "mq_epilogue.h"
#include "mq_default_epilogue_tensor_op.h"

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB_INT4 = cutlass::uint4b_t;
using ElementB_INT8 = int8_t;

using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = int;
// using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
using ElementOutput = cutlass::half_t;
using LayoutOutput = cutlass::layout::ColumnMajor;
using ElementScale = cutlass::half_t;
using ElementZero = uint8_t;
struct Options {
  cutlass::gemm::GemmCoord problem_size;

  int partial_n_int4;
  int partial_n_int8;
  
  Options(int m, int n, int k, int partial_n_int8):
    problem_size({m, n, k}){
    this->partial_n_int8 = partial_n_int8;
    this->partial_n_int4 = n - partial_n_int8;
    // std::cout << "\n[mix] M = " << problem_size.m() << ", N = " << problem_size.n() << ", K = " << problem_size.k() 
    //       << ", partial_n_int4 = " << this->partial_n_int4
    //       << ", partial_n_int8 = " << this->partial_n_int8 << std::endl;
  }
};

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static void __checkCudaErrors(cudaError_t err, const char *filename, int line) {
  assert(filename);
  if (err != cudaSuccess) {
    std::cerr << "[" << filename << ":" << line << "] failed: " << cudaGetErrorString(err) << std::endl;
  }
}

namespace extension_cpp {

////////////////////////////////////////////////////////////////////////////////

template <typename Mma, typename SharedStorage, typename Epilogue = void>
__global__ void kernel_multistage_mma(cutlass::gemm::GemmCoord problem_size,
                                      typename Mma::IteratorA::Params params_A,
                                      ElementA *ptr_A,
                                      typename Mma::IteratorB::Params params_B,
                                      typename Mma::IteratorB::Element *ptr_B,
                                      typename Mma::IteratorScale::Params params_scale,
                                      ElementScale *ptr_scale,
                                      typename Mma::IteratorScaleAct::Params params_scale_act,
                                      ElementScale *ptr_scale_act,
                                      typename Mma::IteratorZero::Params params_zero,
                                      ElementZero *ptr_zero,
                                      ElementOutput *ptr_C, 
                                      typename Mma::LayoutC::Stride::Index ldc,
                                      ElementC *indices) {
  // Shared storage needed by threadblock-scoped matrix multiply-accumulate

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  SharedStorage *shared_storage =
      reinterpret_cast<SharedStorage *>(SharedStorageBase);

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {int(blockIdx.x), int(blockIdx.y),
                                             0};

  cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                   tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                   tb_tile_offset.n() * Mma::Shape::kN};


  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);

  // Output results
  cutlass::MatrixCoord tile_offset = 
        cutlass::make_Coord((tb_tile_offset.m() * Mma::WarpCount::kM) + (warp_id % Mma::WarpCount::kM),
                            (tb_tile_offset.n() * Mma::WarpCount::kN) + (warp_id / Mma::WarpCount::kM)) 
      * cutlass::make_Coord(Mma::Operator::Shape::kM, Mma::Operator::Shape::kN);

  int quad = (threadIdx.x >> 2);
  int lane_in_quad = (threadIdx.x & 3);

  const int kElementsPerAccess = Mma::Operator::InstructionShape::kN / 4;
  const int kRowsPerTile = 8;
  const int kAccumulatorRows = Mma::Operator::InstructionShape::kM / kRowsPerTile;

  cutlass::MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
  
  cutlass::MatrixCoord offset = tile_offset + lane_offset;

  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(params_A, ptr_A,
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);

  typename Mma::IteratorB iterator_B(params_B, ptr_B,
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  typename cutlass::MatrixCoord::Index scale_row_extent = problem_size.k() / 128;
  typename cutlass::MatrixCoord::Index fg_row_offset = tb_tile_offset.k() * problem_size.k() / 64;
  typename cutlass::MatrixCoord::Index scale_row_offset = fg_row_offset;
  cutlass::MatrixCoord tb_offset_scale{scale_row_offset, tb_tile_offset.n() * Mma::Shape::kN};
  cutlass::MatrixCoord tb_offset_scale_act{scale_row_offset, tb_tile_offset.m() * Mma::Shape::kM};

  typename Mma::IteratorScale iterator_scale(
    params_scale, ptr_scale,
    {scale_row_extent, problem_size.n()},
    tb_thread_id,
    tb_offset_scale,
    128/*group_size*/); 
  typename Mma::IteratorScaleAct iterator_scale_act(
    params_scale_act, ptr_scale_act,
    {scale_row_extent, problem_size.m()},
    tb_thread_id,
    tb_offset_scale_act,
    128/*group_size*/);
  typename Mma::IteratorZero iterator_zero(
    params_zero, ptr_zero,
    {scale_row_extent, problem_size.n()},
    tb_thread_id,
    tb_offset_scale,
    128/*group_size*/); 
  

  // Construct thread-scoped matrix multiply
  Mma mma(shared_storage->main_loop, tb_thread_id, warp_id, threadIdx.x);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, iterator_scale, iterator_scale_act, iterator_zero, accum);
  if constexpr (cutlass::platform::is_same<Epilogue, void>::value){
    static_assert(cutlass::platform::is_same<typename Mma::LayoutC, cutlass::layout::ColumnMajor>::value);
    typename Mma::AccumFragmentConvert *converted_accum 
          = reinterpret_cast<typename Mma::AccumFragmentConvert *>(accum.data());

    using IndiceFragment = cutlass::Array<int, Mma::Operator::MmaIterations::kColumn * kElementsPerAccess>;
    IndiceFragment indicesFrag;
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Mma::Operator::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kElementsPerAccess; ++col) {
        int accum_n = mma_n * Mma::Operator::InstructionShape::kN * Mma::Operator::IteratorC::OpDelta::kColumn + col;
        indicesFrag[mma_n*kElementsPerAccess + col] = indices[offset.column() + accum_n];
      }
    }


    using Layout = cutlass::layout::ColumnMajor;
    using TensorRef = cutlass::TensorRef<ElementOutput, Layout>;
    
    TensorRef offset_ref({ptr_C, ldc});

    cutlass::NumericConverter<ElementOutput, float> convert_to_output; 

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Mma::Operator::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Mma::Operator::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Mma::Operator::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * Mma::Operator::InstructionShape::kM * Mma::Operator::IteratorC::OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * Mma::Operator::InstructionShape::kN * Mma::Operator::IteratorC::OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            
            if(offset.row() + accum_m < problem_size.m() && offset.column() + accum_n < problem_size.n()){
              offset_ref.at({offset.row() + accum_m, indicesFrag[mma_n*kElementsPerAccess + col]})
                = convert_to_output((*converted_accum)[idx]);
            }
          }
        }
      }
    }
  }
  else{
    static_assert(cutlass::platform::is_same<typename Mma::LayoutC, cutlass::layout::RowMajor>::value);
    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator::Params params_C(ldc);
    cutlass::MatrixCoord threadblock_offset(
      tb_tile_offset.m() * Mma::Shape::kM,
      tb_tile_offset.n() * Mma::Shape::kN
    );
    
    // Tile iterator writing to destination tensor.
    // TODO: Add threadblock offset in the 5th parameter !!!!!!!!!!!!!!!!!!!!
    typename Epilogue::OutputTileIterator iterator_C(
      params_C,
      ptr_C,
      problem_size.mn(),
      tb_thread_id,
      threadblock_offset
    );
    Epilogue epilogue(
      shared_storage->epilogue,
      tb_thread_id,
      warp_id,
      threadIdx.x % 32);
    typename Epilogue::OutputOp::Params params_output_op = typename Epilogue::OutputOp::Params();
    // Execute the epilogue operator to update the destination tensor.
    typename Epilogue::OutputOp output_op(params_output_op);
    epilogue(
      output_op,
      iterator_C,
      accum);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_,
    bool UseEpilogue = false>
struct Testbed;

/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_>
struct Testbed <MmaCore_, false> {
  /// Threadblock-level GEMM implementation
  using MmaCore = MmaCore_;
  using ThreadblockShape = typename MmaCore::Shape;
  using WarpShape = typename MmaCore::WarpShape;
  using InstructionShape = typename MmaCore::InstructionShape;
  using ElementA = typename MmaCore::ElementA;
  using LayoutA = typename MmaCore::LayoutA;
  using ElementB = typename MmaCore::ElementB;
  using LayoutB = typename MmaCore::LayoutB;
  using ElementC = typename MmaCore::ElementC;
  using LayoutC = typename MmaCore::LayoutC;
  
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  static int const Stages = MmaCore::kStages;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      MmaCore::kCacheOpA;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      MmaCore::kCacheOpB;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using Mma = cutlass::gemm::threadblock::MQMmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      CacheOpA, IteratorB, typename MmaCore::SmemIteratorB, CacheOpB, ElementC,
      LayoutC, typename MmaCore::MmaPolicy, Stages>;

    /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
  };

  using ElementScale = typename Mma::ElementScale;
  using ElementZero = typename Mma::ElementZero;
  using IteratorScale = typename Mma::IteratorScale;
  using IteratorScaleAct = typename Mma::IteratorScaleAct;
  using IteratorZero = typename Mma::IteratorZero;

  int smem_size;
  cudaEvent_t local_gemm_event;

  /// Allocates workspace in device memory
  Testbed(){
      this->smem_size = set_shared_memory();
  }

  ~Testbed(){
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    //
    // Determine SMEM requirements and waive if not satisfied
    //

    cudaDeviceProp properties;
    int device_idx;
    checkCudaErrors(cudaGetDevice(&device_idx));

    checkCudaErrors(cudaGetDeviceProperties(&properties, device_idx));

    return true;
  }

  int set_shared_memory(){
    int smem_size = int(sizeof(typename Testbed::SharedStorage));
    if (smem_size >= (48 << 10)) {
      checkCudaErrors(cudaFuncSetAttribute(
          extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      checkCudaErrors(cudaFuncSetAttribute(
          extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    return smem_size;
  }

  /// Runs the test
  void run(
      dim3 grid, dim3 block,
      Options options,
      torch::Tensor &matrix_A,
      torch::Tensor &matrix_B_interleaved,
      torch::Tensor &matrix_scale_act,
      torch::Tensor &matrix_scale,
      torch::Tensor &matrix_zero,
      torch::Tensor &matrix_indices,
      torch::Tensor &matrix_C_computed,
      cudaStream_t &local_stream) {

    cutlass::gemm::GemmCoord problem_size(
          options.problem_size.m(), 
          cutlass::sizeof_bits<ElementB>::value == 4 ? options.partial_n_int4 : options.partial_n_int8,
          options.problem_size.k());

    typename IteratorA::Params params_A(matrix_A.stride(0));
    typename IteratorB::Params params_B(problem_size.k());
    typename IteratorScale::Params params_scale(matrix_scale.stride(0));
    typename IteratorScaleAct::Params params_scale_act(matrix_scale_act.stride(0));
    typename IteratorZero::Params params_zero(matrix_zero.stride(0));

    extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage>
        <<<grid, block, smem_size, local_stream>>>(
            problem_size, 
            params_A, matrix_A.data_ptr<ElementA>(),
            params_B, 
            reinterpret_cast<ElementB*>(matrix_B_interleaved.data_ptr()),
            params_scale, 
            reinterpret_cast<ElementScale*>(matrix_scale.data_ptr<at::Half>()),
            params_scale_act, 
            reinterpret_cast<ElementScale*>(matrix_scale_act.data_ptr<at::Half>()),
            params_zero, matrix_zero.data_ptr<ElementZero>(),
            reinterpret_cast<ElementOutput*>(matrix_C_computed.data_ptr()),
            matrix_C_computed.stride(0),
            matrix_indices.data_ptr<ElementC>());
  }
};


/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_>
struct Testbed <MmaCore_, true> {
  /// Threadblock-level GEMM implementation
  using MmaCore = MmaCore_;
  using ThreadblockShape = typename MmaCore::Shape;
  using WarpShape = typename MmaCore::WarpShape;
  using InstructionShape = typename MmaCore::InstructionShape;
  using ElementA = typename MmaCore::ElementA;
  using LayoutA = typename MmaCore::LayoutA;
  using ElementB = typename MmaCore::ElementB;
  using LayoutB = typename MmaCore::LayoutB;
  using ElementC = typename MmaCore::ElementC;
  using LayoutC = typename MmaCore::LayoutC;
  
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  static int const Stages = MmaCore::kStages;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      MmaCore::kCacheOpA;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      MmaCore::kCacheOpB;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using Mma = cutlass::gemm::threadblock::MQMmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      CacheOpA, IteratorB, typename MmaCore::SmemIteratorB, CacheOpB, ElementC,
      LayoutC, typename MmaCore::MmaPolicy, Stages>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  /// Define the epilogue
  using EpilogueOutputOp = cutlass::epilogue::thread::Convert<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator>;
  using Epilogue =
      typename cutlass::epilogue::threadblock::MQDefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  using ElementScale = typename Mma::ElementScale;
  using ElementZero = typename Mma::ElementZero;
  using IteratorScale = typename Mma::IteratorScale;
  using IteratorScaleAct = typename Mma::IteratorScaleAct;
  using IteratorZero = typename Mma::IteratorZero;

  int smem_size;
  cudaEvent_t local_gemm_event;

  /// Allocates workspace in device memory
  Testbed(){
      this->smem_size = set_shared_memory();
  }

  ~Testbed(){
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    //
    // Determine SMEM requirements and waive if not satisfied
    //

    cudaDeviceProp properties;
    int device_idx;
    checkCudaErrors(cudaGetDevice(&device_idx));

    checkCudaErrors(cudaGetDeviceProperties(&properties, device_idx));

    return true;
  }

  int set_shared_memory(){
    int smem_size = int(sizeof(typename Testbed::SharedStorage));
    if (smem_size >= (48 << 10)) {
      checkCudaErrors(cudaFuncSetAttribute(
          extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage, Epilogue>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      checkCudaErrors(cudaFuncSetAttribute(
          extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage, Epilogue>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    return smem_size;
  }

  /// Runs the test
  void run(
      dim3 grid, dim3 block,
      Options options,
      torch::Tensor &matrix_A,
      torch::Tensor &matrix_B_interleaved,
      torch::Tensor &matrix_scale_act,
      torch::Tensor &matrix_scale,
      torch::Tensor &matrix_zero,
      torch::Tensor &matrix_indices,
      torch::Tensor &matrix_C_computed,
      cudaStream_t &local_stream) {

    cutlass::gemm::GemmCoord problem_size(
          options.problem_size.m(), 
          cutlass::sizeof_bits<ElementB>::value == 4 ? options.partial_n_int4 : options.partial_n_int8,
          options.problem_size.k());

    typename IteratorA::Params params_A(matrix_A.stride(0));
    typename IteratorB::Params params_B(problem_size.k());
    typename IteratorScale::Params params_scale(matrix_scale.stride(0));
    typename IteratorScaleAct::Params params_scale_act(matrix_scale_act.stride(0));
    typename IteratorZero::Params params_zero(matrix_zero.stride(0));

    extension_cpp::kernel_multistage_mma<Mma, Testbed::SharedStorage, Epilogue>
        <<<grid, block, smem_size, local_stream>>>(
            problem_size, 
            params_A, matrix_A.data_ptr<ElementA>(),
            params_B, 
            reinterpret_cast<ElementB*>(matrix_B_interleaved.data_ptr()),
            params_scale, 
            reinterpret_cast<ElementScale*>(matrix_scale.data_ptr<at::Half>()),
            params_scale_act, 
            reinterpret_cast<ElementScale*>(matrix_scale_act.data_ptr<at::Half>()),
            params_zero, matrix_zero.data_ptr<ElementZero>(),
            reinterpret_cast<ElementOutput*>(matrix_C_computed.data_ptr()),
            matrix_C_computed.stride(0),
            matrix_indices.data_ptr<ElementC>());
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
