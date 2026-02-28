/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#if !defined(__CUDACC_RTC__)
#include <cfenv>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/thread/unary_op.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/array.h"
#include "cutlass/half.h"

namespace cutlass {

/// Partial specialization for Array<int8_t, 8> <= Array<uint4b_t, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, uint4b_t, 8, Round> {
  using result_type = Array<int8_t, 8>;
  using source_type = Array<uint4b_t, 8>;
  static FloatRoundStyle const round_style = Round;
  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    unsigned const& storage = reinterpret_cast<unsigned const &>(source);
    unsigned out[2];
    asm volatile(
        "{ .reg .u32 tmp0;"
        "and.b32 tmp0, %2, 0xf0f0f0f0;"
        "and.b32 %0, %2, 0x0f0f0f0f;"
        "shr.u32 %1, tmp0, 4;"
        "}"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(storage));
    return reinterpret_cast<result_type const &>(out);
  }
  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<uint4b_t>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, uint4b_t, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");
  using result_type = Array<int8_t, N>;
  using source_type = Array<uint4b_t, N>;
  static FloatRoundStyle const round_style = Round;
  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayConverter<int8_t, uint4b_t, 8, Round> convert_vector_;
    result_type result;
    Array<int8_t, 8> *result_ptr = reinterpret_cast<Array<int8_t, 8> *>(&result);
    Array<uint4b_t, 8> const *source_ptr = reinterpret_cast<Array<uint4b_t, 8> const *>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }
    return result;
  }
  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <int N, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct FastInt2FloatNumericArrayConverterEpilogue {
  using result_type = Array<float, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = reinterpret_cast<float const &>(source[i]) - 12582912.0f;
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};


}
