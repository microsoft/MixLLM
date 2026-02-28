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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once


#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/arch/memory_sm75.h"

#include "mq_mma_base.h"
#include "mq_mma_tensor_op_dequantizer.h"
#include "mq_fine_grained_scale_zero_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename MmaShape, typename Element, typename Layout, int kAlignment, bool IsAct>
struct DefaultScaleIteratorsMultistage;
// Fine grained iterators on weight
template <typename MmaShape, typename Element, typename Layout, int kAlignment>
struct DefaultScaleIteratorsMultistage<MmaShape, Element, Layout, kAlignment, false>
{
    using Iterator
        = cutlass::transform::threadblock::FineGrainedScaleZeroIterator<cutlass::MatrixShape<1, MmaShape::kN>, 
            Element, kAlignment, Layout>;
    using SmemIterator = Iterator;
};

// Fine grained iterators on activation
template <typename MmaShape, typename Element, typename Layout, int kAlignment>
struct DefaultScaleIteratorsMultistage<MmaShape, Element, Layout, kAlignment, true>
{
    using Iterator
        = cutlass::transform::threadblock::FineGrainedScaleZeroIterator<cutlass::MatrixShape<1, MmaShape::kM>, 
            Element, kAlignment, Layout>;
    using SmemIterator = Iterator;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class MQMmaMultistage : 
  public MQMmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MQMmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using ElementScale = typename Base::ElementScale;
  using ElementZero = typename Base::ElementZero;
  using ScaleIterators = DefaultScaleIteratorsMultistage<Shape, ElementScale, layout::RowMajor, 2, false>;
  using ScaleIteratorsAct = DefaultScaleIteratorsMultistage<Shape, ElementScale, layout::RowMajor, 2, true>;
  // Define iterators over tiles from the scale operand
  using IteratorScale = typename ScaleIterators::Iterator;
  using IteratorScaleAct = typename ScaleIteratorsAct::Iterator;
  using SmemIteratorScale = typename ScaleIterators::SmemIterator;
  using SmemIteratorScaleAct = typename ScaleIteratorsAct::SmemIterator;
  using LayoutScale = typename IteratorScale::Layout;

  using ZeroIterators  = DefaultScaleIteratorsMultistage<Shape, ElementZero, layout::RowMajor, 4, false>;
  using IteratorZero = typename ZeroIterators::Iterator;
  using SmemIteratorZero = typename ZeroIterators::SmemIterator;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  constexpr static bool is_int8_x_int8 = 
            std::is_same<typename Operator::FragmentA::Element, int8_t>::value 
         && std::is_same<typename Operator::FragmentB::Element, int8_t>::value;
  using Dequantizer = warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, ElementScale,
    LayoutScale, 32, !is_int8_x_int8, false>;
  using DequantizerAct = warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, ElementScale,
  LayoutScale, 32, false, true>;

  using ElementConvert = float;
  using AccumFragmentConvert = cutlass::Array<ElementConvert, FragmentC::kElements>;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:
  static bool constexpr post_scaling = true;

  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];

    typename Dequantizer::FragmentScale warp_frag_scales;
    typename Dequantizer::FragmentZero warp_frag_zeros;
    typename DequantizerAct::FragmentScale warp_frag_scales_act;
  };


 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Iterator to write threadblock-scoped tile of scale/zero operand to shared memory
  SmemIteratorScale smem_iterator_scale_;
  SmemIteratorScaleAct smem_iterator_scale_act_;
  SmemIteratorZero smem_iterator_zero_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

  Dequantizer warp_dequantizer_;
  DequantizerAct warp_dequantizer_act_;


public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MQMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      warp_dequantizer_({shared_storage.operand_scale.data(), LayoutScale(Shape::kN)},
        {shared_storage.operand_zero.data(), LayoutScale(Shape::kN)},
        (warp_idx % (Base::WarpCount::kN * Base::WarpCount::kM)) / Base::WarpCount::kM, lane_idx),
      warp_dequantizer_act_({shared_storage.operand_scale_act.data(), LayoutScale(Shape::kM)},
        {nullptr, LayoutScale(Shape::kM)},
        (warp_idx % (Base::WarpCount::kN * Base::WarpCount::kM)) % Base::WarpCount::kM, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_iterator_scale_(
        LayoutScale(Shape::kN), 
        shared_storage.operand_scale.data(), 
        {Base::kStages, Shape::kN},
        thread_idx,
        128 /*group size*/),
      smem_iterator_scale_act_(
        LayoutScale(Shape::kM), 
        shared_storage.operand_scale_act.data(), 
        {Base::kStages, Shape::kM},
        thread_idx,
        128 /*group size*/),
      smem_iterator_zero_(
        LayoutScale(Shape::kN), 
        shared_storage.operand_zero.data(), 
        {Base::kStages, Shape::kN},
        thread_idx,
        128 /*group size*/),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    
    if constexpr (post_scaling)
      static_assert(Detail::kStagedAccumulation == false, 
        "Does not support staged accumulation for the quant kernel.");

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    if (smem_read_stage_idx_ == Base::kStages - 1) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_dequantizer_.add_pointer_offset((1-Base::kStages) * Shape::kN);
      this->warp_dequantizer_act_.add_pointer_offset((1-Base::kStages) * Shape::kM);
      smem_read_stage_idx_ = 0;
    }
    else{
      ++smem_read_stage_idx_;
      warp_dequantizer_.add_pointer_offset(Shape::kN);
      warp_dequantizer_act_.add_pointer_offset(Shape::kM);
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA &iterator_A,
    IteratorB &iterator_B)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_iterator_scale_.add_tile_offset({-Base::kStages, 0});
      smem_iterator_scale_act_.add_tile_offset({-Base::kStages, 0});
      smem_iterator_zero_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

    CUTLASS_DEVICE
  void copy_scales_and_advance(IteratorScale& iterator_scale, SmemIteratorScale& smem_iterator_scale,
                               IteratorScaleAct& iterator_scale_act, SmemIteratorScaleAct& smem_iterator_scale_act,
                               IteratorZero& iterator_zero, SmemIteratorZero& smem_iterator_zero)
  {
      static_assert(IteratorScale::Shape::kRow == 1, "Scale/zero's row must be 1.");

      typename IteratorScale::AccessType* smem_scale_ptr
          = reinterpret_cast<typename IteratorScale::AccessType*>(smem_iterator_scale.get());
      typename IteratorScaleAct::AccessType* smem_scale_act_ptr
          = reinterpret_cast<typename IteratorScaleAct::AccessType*>(smem_iterator_scale_act.get());
      typename IteratorZero::AccessType* smem_zero_ptr
          = reinterpret_cast<typename IteratorZero::AccessType*>(smem_iterator_zero.get());

      int const kSrcBytes = sizeof_bits<typename IteratorScale::Element>::value * IteratorScale::kAlignment / 8;
      static_assert(kSrcBytes == sizeof_bits<typename IteratorZero::Element>::value * IteratorZero::kAlignment / 8);

      typename IteratorScale::AccessType* gmem_scale_ptr = iterator_scale.get();
      typename IteratorScaleAct::AccessType* gmem_scale_act_ptr = iterator_scale_act.get();
      typename IteratorZero::AccessType* gmem_zero_ptr = iterator_zero.get();

      // cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(smem_scale_ptr, gmem_scale_ptr, iterator_scale.valid());
      // cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(smem_scale_act_ptr, gmem_scale_act_ptr, iterator_scale_act.valid());

      static_assert(Shape::kK == 64);
      if (iterator_scale.row_groupsize64_ & 0x1)
      {
        iterator_scale.add_tile_offset({1, 0});
        iterator_scale_act.add_tile_offset({1, 0});
        iterator_zero.add_tile_offset({1, 0});
      }
      else{
        // if (iterator_scale.valid()) {
        //   *smem_scale_ptr = *gmem_scale_ptr;
        // }
        
        // if (iterator_scale_act.valid()) {
        //   *smem_scale_act_ptr = *gmem_scale_act_ptr;
        // }
              
        unsigned smem_scale_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_scale_ptr);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
            "}\n" ::"r"((int)iterator_scale.valid()),
            "r"(smem_scale_int_ptr), "l"(gmem_scale_ptr), "n"(kSrcBytes));

        unsigned smem_scale_act_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_scale_act_ptr);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
            "}\n" ::"r"((int)iterator_scale_act.valid()),
            "r"(smem_scale_act_int_ptr), "l"(gmem_scale_act_ptr), "n"(kSrcBytes));

        if constexpr (!is_int8_x_int8){
          unsigned smem_zero_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_zero_ptr);
          asm volatile(
              "{\n"
              "  .reg .pred p;\n"
              "  setp.ne.b32 p, %0, 0;\n"
              "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
              "}\n" ::"r"((int)iterator_zero.valid()),
              "r"(smem_zero_int_ptr), "l"(gmem_zero_ptr), "n"(kSrcBytes));
        }
      }
      iterator_scale.row_groupsize64_++;
      smem_iterator_scale.add_tile_offset({1, 0});
      smem_iterator_scale_act.add_tile_offset({1, 0});
      smem_iterator_zero.add_tile_offset({1, 0});
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    IteratorScale &iterator_scale,
    IteratorScaleAct &iterator_scale_act,
    IteratorZero &iterator_zero,
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);
      iterator_scale.clear_mask(gemm_k_iterations == 0);
      iterator_scale_act.clear_mask(gemm_k_iterations == 0);
      iterator_zero.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }
      copy_scales_and_advance(iterator_scale, this->smem_iterator_scale_, 
                              iterator_scale_act, this->smem_iterator_scale_act_,
                              iterator_zero, this->smem_iterator_zero_);
      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }
  }


  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }


  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    IteratorScale &iterator_scale,
    IteratorScaleAct &iterator_scale_act,
    IteratorZero &iterator_zero,
    int &gemm_k_iterations)         ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragment from shared memory
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }
      if constexpr (!is_int8_x_int8)
        warp_dequantizer_.apply_zero(pipe_state.warp_transformed_frag_B_[warp_mma_k % 2], pipe_state.warp_frag_zeros);


      // Execute the current warp-tile of MMA operations
      if constexpr (Detail::kStagedAccumulation || post_scaling) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0 && !post_scaling) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
            iterator_A,
            iterator_B,
            group_start_iteration_A,
            group_start_iteration_B);

        if (group_start_iteration_B == 0)
        {
          copy_scales_and_advance(iterator_scale, this->smem_iterator_scale_,
                                  iterator_scale_act, this->smem_iterator_scale_act_,
                                  iterator_zero, this->smem_iterator_zero_);
        }
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {

        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
          iterator_A,
          iterator_B,
          group_start_iteration_A,
          group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
        iterator_scale.clear_mask(gemm_k_iterations == 0);
        iterator_scale_act.clear_mask(gemm_k_iterations == 0);
        iterator_zero.clear_mask(gemm_k_iterations == 0);
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {

        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }

    }
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      IteratorB &iterator_B,
      IteratorScale &iterator_scale,
      IteratorScaleAct &iterator_scale_act,
      IteratorZero &iterator_zero)        ///< [in|out] iterator over B operand in global memory
  {
    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);
    iterator_scale.clear_mask(gemm_k_iterations == 0);
    iterator_scale_act.clear_mask(gemm_k_iterations == 0);
    iterator_zero.clear_mask(gemm_k_iterations == 0);

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    warp_dequantizer_.load(pipe_state.warp_frag_scales, pipe_state.warp_frag_zeros);
    warp_dequantizer_act_.load(pipe_state.warp_frag_scales_act);


    // Transform, if necessary, the first warp-tile's shared memory fragments
    warp_mma_.transform(
      pipe_state.warp_transformed_frag_A_[0],
      pipe_state.warp_transformed_frag_B_[0],
      pipe_state.warp_loaded_frag_A_[0],
      pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    // Mainloop
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 2);) {
      pipe_state.tmp_accum_.fill(1262485504);

      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A,
        iterator_B,
        iterator_scale,
        iterator_scale_act,
        iterator_zero,
        gemm_k_iterations);


      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A,
        iterator_B,
        iterator_scale,
        iterator_scale_act,
        iterator_zero,
        gemm_k_iterations);

      warp_dequantizer_.apply_scale_accum(pipe_state.tmp_accum_, pipe_state.warp_frag_scales);
      warp_dequantizer_act_.apply_scale_accum_act(pipe_state.tmp_accum_, pipe_state.warp_frag_scales_act);
      
      warp_dequantizer_.load(pipe_state.warp_frag_scales, pipe_state.warp_frag_zeros);
      warp_dequantizer_act_.load(pipe_state.warp_frag_scales_act);


      AccumFragmentConvert* converted_accum = reinterpret_cast<AccumFragmentConvert*>(accum.data());
      AccumFragmentConvert* converted_tmp_accum = reinterpret_cast<AccumFragmentConvert*>(pipe_state.tmp_accum_.data());

      // FastNumericArrayConverter<ElementConvert, typename FragmentC::Element, FragmentC::kElements> convert_accum_array;
      // *converted_tmp_accum = convert_accum_array(pipe_state.tmp_accum_);

      // CUTLASS_PRAGMA_UNROLL
      // for(int i = 0; i < AccumFragmentConvert::kElements; i ++)
      //   converted_tmp_accum[i] = converted_tmp_accum[i] * 1.0f;

      plus<AccumFragmentConvert> plus_accum;
      *converted_accum = plus_accum(*converted_accum, *converted_tmp_accum);
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA &iterator_A,
      ///< iterator over B operand in global memory
      IteratorB &iterator_B,
      IteratorScale &iterator_scale,
      IteratorScaleAct &iterator_scale_act,
      IteratorZero &iterator_zero,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_B, iterator_scale, iterator_scale_act, iterator_zero, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Initialize destination accumulators with source accumulators
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B, iterator_scale, iterator_scale_act, iterator_zero);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

