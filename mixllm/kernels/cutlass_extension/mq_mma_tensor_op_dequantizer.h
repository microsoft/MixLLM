/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"


////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace warp
{

////////////////////////////////////////////////////////////////////////////////

template <
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of Scale elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    bool hasZero,
    bool applyA = true, 
    ///
    typename Enable = void>
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////

// Specialization for Turing & Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    bool hasZero,
    bool applyA>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, half_t, layout::RowMajor, 32,
    hasZero, applyA, 
    typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 75>::type>
{

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementScale = half_t;
    using ElementZero = uint8_t;
    using ElementConvert = float;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = InstructionShape::kN / 8;
    static constexpr int kRowsPerMmaPerThread = InstructionShape::kM / 8;

    static constexpr int kScaleElements = applyA ? 
                    kRowsPerMmaPerThread * MmaOperator::MmaIterations::kRow : 
                    kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn * 2;
                    
    
    using FragmentScale = Array<ElementScale, kScaleElements>;
    using FragmentZero = Array<ElementZero, MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRefScale = TensorRef<ElementScale, Layout>;
    using TensorRefZero = TensorRef<ElementZero, Layout>;
    
    int threadidx_print = 0;
    int blockidx_print = 0;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRefScale smem_scales, TensorRefZero smem_zeros, int const warp_idx_n, int const lane_idx)
    {
        if constexpr(applyA){
            static_assert(!hasZero, "Activation dequantizer should not has zero.");
            int const warp_offset = warp_idx_n * Shape::kM;
            int const quad = lane_idx / 4;
            int const thread_offset = warp_offset + quad;
            pointer_scale_ = smem_scales.data() + thread_offset;

        }
        else{
            // NumericConverter<int, cutlass::half_t> convert;
            int const warp_offset = warp_idx_n * Shape::kN;
            int const quad = lane_idx % 4;
            int const thread_offset = warp_offset + quad * 2;
            pointer_scale_ = smem_scales.data() + thread_offset;
            if constexpr (hasZero)
                pointer_zero_ = smem_zeros.data() + warp_offset + lane_idx / 4;
            // if(threadIdx.x==0 && threadIdx.y==0){
            //     printf("%d, ", thread_offset);
            //     for(int i = 0; i < 64; i ++)
            //         printf("%d ", convert(pointer_scale_[i]));
            //     printf("\n");
            // }
        }
    }

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRefScale smem_scales, int const warp_idx_n, int const lane_idx)
        : MmaTensorOpDequantizer(smem_scales, TensorRefZero(), warp_idx_n, lane_idx)
    {
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        static_assert(applyA, "This load should only apply on activation scale");

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kScaleElements; mma_n_iter++)
        {
            scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
        }
    }


    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag, FragmentZero& zero_frag)
    {
        static_assert(!applyA, "This load should only apply on weight scale");

        using ExpandedScale
            = Array<ElementScale, InstructionShape::kN>;
        const ExpandedScale* pointer_scale_ptr = reinterpret_cast<const ExpandedScale*>(pointer_scale_);

        half2* scale_frag_ptr = reinterpret_cast<half2*>(scale_frag.data());

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kScaleElements/2; ++mma_n_iter)
        {
            const half2* scale_ptr = reinterpret_cast<const half2*>(&pointer_scale_ptr[mma_n_iter]);
            scale_frag_ptr[mma_n_iter] = scale_ptr[0];
        }

        if constexpr (hasZero){
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; mma_n_iter++)
            {
                zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
            }
        }
    }


    CUTLASS_DEVICE
    void apply_zero(typename MmaOperator::TransformedFragmentB& operand_frag, FragmentZero const& zero_frag)
    {
        using MmaOperandB = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename MmaOperandB::Element, MmaOperandB::kElements>;
        static_assert(MmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                == MmaOperator::TransformedFragmentB::kElements,
            "");

        // 0,0,0,0,0,0,0,0
        // 1,1,1,1,1,1,1,1
        // 2,2,2,2,2,2,2,2
        // 3,3,3,3,3,3,3,3
        
        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        static_assert(FragmentZero::kElements == MmaOperator::MmaIterations::kColumn);

        if constexpr (FragmentZero::kElements % 4 != 0){
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m_iter = 0; mma_m_iter < MmaOperator::MmaIterations::kColumn; ++mma_m_iter)
            {
                typename MmaOperandB::Element* operand_ptr = reinterpret_cast<typename MmaOperandB::Element*>(&operand_frag_ptr[mma_m_iter]);
                CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < MmaOperandB::kElements; ++ii){
                    operand_ptr[ii] = operand_ptr[ii] - zero_frag[mma_m_iter];
                }
            }
        }
        else{
            uint32_t const* packed_zeros_ptr = reinterpret_cast<uint32_t const*>(zero_frag.data());
            uint32_t zero_points[4];

            CUTLASS_PRAGMA_UNROLL
            for(int outer_index = 0; outer_index < MmaOperator::MmaIterations::kColumn/4; outer_index ++){
                uint32_t const packed_zeros = packed_zeros_ptr[outer_index];
                zero_points[0] = __byte_perm(packed_zeros, 0, 0x00000000);
                zero_points[1] = __byte_perm(packed_zeros, 0, 0x00001111);
                zero_points[2] = __byte_perm(packed_zeros, 0, 0x00002222);
                zero_points[3] = __byte_perm(packed_zeros, 0, 0x00003333);

                CUTLASS_PRAGMA_UNROLL
                for (int mma_m_iter = 0; mma_m_iter < 4; ++mma_m_iter){
                    uint32_t* operand_ptr = reinterpret_cast<uint32_t*>(&operand_frag_ptr[mma_m_iter + outer_index*4]);
                    CUTLASS_PRAGMA_UNROLL
                    for (int ii = 0; ii < 2; ++ii){
                        // operand_ptr[ii] = __vsub4(operand_ptr[ii], zero_points[mma_m_iter]);
                        asm volatile("vsub4.u32.u32.u32 %0,%1,%2,%3;"
                            : "=r"(operand_ptr[ii])
                            : "r"(operand_ptr[ii]), "r"(zero_points[mma_m_iter]), "r"(0));
                    }
                }
            }
        }

    }

    CUTLASS_DEVICE
    void apply_scale_accum_act(typename MmaOperator::FragmentC& accum, FragmentScale const& scale_frag)
    {
        using FragmentC = typename MmaOperator::FragmentC;
        using MmaIterations = typename MmaOperator::MmaIterations;
        using ElementAccum = typename FragmentC::Element;

        static_assert(std::is_same<typename FragmentC::Element, int>::value, 
                      "Accumulator should be int");

        constexpr int accums_in_row_per_mma = MmaIterations::kRow * kRowsPerMmaPerThread * 2;

        using ConvertedAccumRowFragment = Array<ElementConvert, accums_in_row_per_mma>;
        using ConvertedScaleFragment = Array<ElementConvert, FragmentScale::kElements>;

        ConvertedAccumRowFragment* converted_accum_row_fragment_ptr = \
            reinterpret_cast<ConvertedAccumRowFragment*>(accum.data());
        
        NumericConverter<ElementConvert, ElementScale> convert_scale;

        
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
        // 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn; ++m) {
            ElementConvert* converted_accum_inner_ptr = \
                reinterpret_cast<ElementConvert*>(&converted_accum_row_fragment_ptr[m]);

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < accums_in_row_per_mma; ++n){
                converted_accum_inner_ptr[n] = \
                    converted_accum_inner_ptr[n] * convert_scale(scale_frag[n/2]);
            }
        }
    }


    CUTLASS_DEVICE
    void apply_scale_accum(typename MmaOperator::FragmentC& accum, FragmentScale const& scale_frag)
    {
        static_assert(std::is_same<typename MmaOperator::FragmentC::Element, int>::value, 
                      "Accumulator should be int");

        using MmaIterations = typename MmaOperator::MmaIterations;
        using FragmentC = typename MmaOperator::FragmentC;
        constexpr int accums_in_row_per_mma = MmaIterations::kRow * kRowsPerMmaPerThread * 2;
        using ConvertedAccumRowFragment = Array<ElementConvert, accums_in_row_per_mma>;
        using ElementAccum = typename FragmentC::Element;

        NumericConverter<ElementConvert, ElementScale> convert_scale;

        // FastNumericArrayConverter<ElementConvert, ElementAccum, FragmentC::kElements> convert_accum_array;
        FastInt2FloatNumericArrayConverterEpilogue<FragmentC::kElements> convert_accum_array;

        using AccumFragmentConvert = cutlass::Array<ElementConvert, FragmentC::kElements>;
        AccumFragmentConvert* converted_accum = reinterpret_cast<AccumFragmentConvert*>(accum.data());
        *converted_accum = convert_accum_array(accum);

        ConvertedAccumRowFragment* converted_accum_row_fragment_ptr 
            = reinterpret_cast<ConvertedAccumRowFragment*>(accum.data());

        // Apply on C, and C is NxM row-major
        // 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
        // 2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,
        // 4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,
        // 6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,
        // 8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,
        // 10,11,10,11,10,11,10,11,10,11,10,11,
        // ...
        // 14,15,14,15,14,15,14,15,14,15,14,15,

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn; ++m) {
            ElementConvert* converted_accum_inner_ptr 
                = reinterpret_cast<ElementConvert*>(&converted_accum_row_fragment_ptr[m]);

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kRow * kRowsPerMmaPerThread * 2; ++n){
                converted_accum_inner_ptr[n] 
                    = converted_accum_inner_ptr[n] * convert_scale(scale_frag[m*2 + n%2]);

            }
        }
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset)
    {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_scale_ += offset;
        pointer_zero_ += offset;
    }

private:
    ElementScale const* pointer_scale_;
    ElementZero const* pointer_zero_;
};


////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////