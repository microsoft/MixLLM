# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch

__all__ = ["quantize", "transpose", "mixllm_gemm"]


def transpose(a):
    return torch.ops.kernels_mixllm.transpose(a)


def quantize(a):
    return torch.ops.kernels_mixllm.quantize(a)


def mixllm_gemm(a, scale_act, zero, scale_int8, scale_int4, indices_int8,
                indices_int4, b_int8, b_int4):
    return torch.ops.kernels_mixllm.gemm(a, scale_act, zero, scale_int8,
                                         scale_int4, indices_int8, indices_int4,
                                         b_int8, b_int4)


@torch.library.register_fake("kernels_mixllm::quantize")
def quantize_abstract(a):
    torch._check(a.dim() == 2, "Input must be a 2D tensor")
    m = a.shape[0]
    n = a.shape[1]
    group_size = 128
    torch._check(a.is_cuda, "Input must be on CUDA device")
    torch._check(a.dtype == torch.float16, "Input must be float16")
    torch._check(
        n % group_size == 0,
        "Input must have a second dimension that is a multiple of group_size")

    m_round_even = m + (m % 2)
    return (torch.empty((m, n), dtype=torch.int8, device="cuda:0"),
            torch.empty((n // group_size, m_round_even),
                        dtype=torch.float16,
                        device="cuda:0"))


@torch.library.register_fake("kernels_mixllm::transpose")
def transpose_abstract(a):
    torch._check(a.dim() == 2, "Input must be a 2D tensor")
    m = a.shape[0]
    n = a.shape[1]
    torch._check(a.is_cuda, "Input must be on CUDA device")
    torch._check(a.dtype == torch.float16, "Input must be float16")
    return torch.empty((n, m), dtype=torch.float16, device="cuda:0")


@torch.library.register_fake("kernels_mixllm::gemm")
def mixllm_gemm_abstract(a, scale_act, zero, scale_int8, scale_int4,
                         indices_int8, indices_int4, b_int8, b_int4):
    torch._check(a.is_cuda, "Input tensor A must be on CUDA device")
    torch._check(a.dtype == torch.int8, "Input tensor A must be int8")
    torch._check(scale_act.dtype == torch.float16,
                 "Scale activation tensor must be float16")
    torch._check(zero.dtype == torch.uint8, "Zero tensor must be uint8")
    torch._check(scale_int8.dtype == torch.float16,
                 "Scale int8 tensor must be float16")
    torch._check(scale_int4.dtype == torch.float16,
                 "Scale int4 tensor must be float16")
    torch._check(indices_int8.dtype == torch.int32,
                 "Indices int8 tensor must be int32")
    torch._check(indices_int4.dtype == torch.int32,
                 "Indices int4 tensor must be int32")
    torch._check(b_int8.dtype == torch.int8, "B int8 tensor must be int8")
    torch._check(b_int4.dtype == torch.uint8, "B int4 tensor must be uint8")
    torch._check(b_int8.is_cuda, "B int8 tensor must be on CUDA device")
    torch._check(b_int4.is_cuda, "B int4 tensor must be on CUDA device")
    torch._check(a.dim() == 2, "Input tensor A must be 2D")
    torch._check(scale_act.dim() == 2, "Scale activation tensor must be 2D")
    torch._check(zero.dim() == 2, "Zero tensor must be 2D")
    torch._check(scale_int8.dim() == 2, "Scale int8 tensor must be 2D")
    torch._check(scale_int4.dim() == 2, "Scale int4 tensor must be 2D")
    torch._check(indices_int8.dim() == 1, "Indices int8 tensor must be 1D")
    torch._check(indices_int4.dim() == 1, "Indices int4 tensor must be 1D")
    torch._check(b_int8.dim() == 2, "B int8 tensor must be 2D")
    torch._check(b_int4.dim() == 2, "B int4 tensor must be 2D")

    m = a.shape[0]
    n = (indices_int4.numel() + indices_int8.numel())
    is_row_major = (indices_int4.numel() == 0 or indices_int8.numel() == 0)
    if is_row_major:
        c = torch.empty((m, n), dtype=torch.float16, device="cuda:0")
    else:
        c = torch.empty((n, m), dtype=torch.float16, device="cuda:0")

    return c
