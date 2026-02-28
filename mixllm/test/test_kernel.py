# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import mixllm
import argparse
from mixllm.nn.modules.utils import bcolors, time_logger

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=32)
parser.add_argument('-n', type=int, default=1024)
parser.add_argument('-k', type=int, default=2048)
parser.add_argument('-r', type=float, default=0.25)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--warm_up', type=int, default=10)
args = parser.parse_args()
print(args)


def down_size_(size, scale):
    assert size[-1] % scale == 0, f"{size} last dim not divisible by {scale}"
    return (*size[:-1], size[-1] // scale)


def pack_uint4_for_cutlass(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size_(
        shape, 2)).contiguous()
    # return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size_(shape, 2)).contiguous()


def interleave_uint4_for_cutlass(matrix_B_int4) -> torch.Tensor:
    matrix_B_interleaved = matrix_B_int4.clone()

    interleaved_index = []
    for i in range(args.k):
        i_sub = i % 32
        if i_sub >= 4 and i_sub < 8:
            interleaved_index.append(i + 12)
        elif i_sub >= 8 and i_sub < 12:
            interleaved_index.append(i - 4)
        elif i_sub >= 12 and i_sub < 16:
            interleaved_index.append(i + 8)
        elif i_sub >= 16 and i_sub < 20:
            interleaved_index.append(i - 8)
        elif i_sub >= 20 and i_sub < 24:
            interleaved_index.append(i + 4)
        elif i_sub >= 24 and i_sub < 28:
            interleaved_index.append(i - 12)
        else:
            interleaved_index.append(i)

    matrix_B_interleaved = matrix_B_int4[:, interleaved_index]
    matrix_B_interleaved_tmp = matrix_B_interleaved.clone()

    interleaved_index = []
    for i in range(0, args.k, 8):
        interleaved_index.append(i)
        interleaved_index.append(i + 4)
        interleaved_index.append(i + 1)
        interleaved_index.append(i + 5)
        interleaved_index.append(i + 2)
        interleaved_index.append(i + 6)
        interleaved_index.append(i + 3)
        interleaved_index.append(i + 7)
    matrix_B_interleaved = matrix_B_interleaved_tmp[:, interleaved_index]

    return pack_uint4_for_cutlass(matrix_B_interleaved)


if __name__ == "__main__":
    int4_start_idx = int(args.n * args.r)
    partial_n_int4 = args.n - int4_start_idx
    partial_n_int8 = int4_start_idx
    min_int8 = -127
    max_int8 = -1 * min_int8 + 1
    groupsize = 128
    is_row_major = (partial_n_int8 == 0 or partial_n_int4 == 0)

    ##############################
    # generate random input data
    ##############################
    matrix_A = torch.randint(min_int8,
                             max_int8, (args.m, args.k),
                             dtype=torch.int8,
                             device="cuda:0")
    matrix_scale_act = 0.1 * torch.randn(
        (args.k // groupsize, ((args.m + 1) // 2) * 2),
        dtype=torch.float16,
        device="cuda:0")
    matrix_zero = torch.randint(0,
                                16, (args.k // groupsize,
                                     (partial_n_int4 + 3) // 4 * 4),
                                dtype=torch.uint8,
                                device="cuda:0")
    matrix_scale_int8 = 0.1 * torch.randn(
        (args.k // groupsize, ((partial_n_int8 + 1) // 2) * 2),
        dtype=torch.float16,
        device="cuda:0")
    matrix_scale_int4 = 0.1 * torch.randn(
        (args.k // groupsize, ((partial_n_int4 + 1) // 2) * 2),
        dtype=torch.float16,
        device="cuda:0")
    matrix_indices = torch.randperm(
        args.n, dtype=torch.int32,
        device="cuda:0") if not is_row_major else torch.arange(
            args.n, dtype=torch.int32, device="cuda:0")
    matrix_indices_int8 = matrix_indices[0:partial_n_int8]
    matrix_indices_int4 = matrix_indices[partial_n_int8:args.n]
    matrix_B_int8 = torch.randint(min_int8,
                                  max_int8, (partial_n_int8, args.k),
                                  dtype=torch.int8,
                                  device="cuda:0")
    matrix_B_int4 = torch.randint(0,
                                  16, (partial_n_int4, args.k),
                                  dtype=torch.uint8,
                                  device="cuda:0")
    matrix_B_interleaved = interleave_uint4_for_cutlass(matrix_B_int4)

    print(f"shape of matrix_A: {matrix_A.shape}")
    print(f"shape of matrix_scale_act: {matrix_scale_act.shape}")
    print(f"shape of matrix_zero: {matrix_zero.shape}")
    print(f"shape of matrix_scale_int8: {matrix_scale_int8.shape}")
    print(f"shape of matrix_scale_int4: {matrix_scale_int4.shape}")
    print(f"shape of matrix_indices_int8: {matrix_indices_int8.shape}")
    print(f"shape of matrix_indices_int4: {matrix_indices_int4.shape}")
    print(f"shape of matrix_B_int8: {matrix_B_int8.shape}")
    print(f"shape of matrix_B_int4: {matrix_B_int4.shape}")
    print(f"shape of matrix_B_interleaved: {matrix_B_interleaved.shape}")

    ##############################
    # run and profile mixllm kernel
    ##############################
    c_mixllm = torch.empty((args.n, args.m),
                           dtype=torch.float16,
                           device="cuda:0")
    if is_row_major:
        c_mixllm = c_mixllm.t_().contiguous()

    c_mixllm = mixllm.nn.modules.ops.mixllm_gemm(
        matrix_A, matrix_scale_act, matrix_zero, matrix_scale_int8,
        matrix_scale_int4, matrix_indices_int8, matrix_indices_int4,
        matrix_B_int8, matrix_B_interleaved)

    cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cudagraph):
        c_mixllm = mixllm.nn.modules.ops.mixllm_gemm(
            matrix_A, matrix_scale_act, matrix_zero, matrix_scale_int8,
            matrix_scale_int4, matrix_indices_int8, matrix_indices_int4,
            matrix_B_int8, matrix_B_interleaved)
    _, latency = time_logger(cudagraph.replay)
    print(bcolors.WARNING + "latency =", "%.2f" % latency, "ms,",
          "%.2f" % (args.m * args.n * args.k * 2 / latency / 1e9),
          "TFLOPs" + bcolors.ENDC)

    if not is_row_major:
        c_mixllm = c_mixllm.t_()

    ##############################
    # check with pytorch reference
    ##############################
    c_int4 = ((matrix_A.float().view(args.m, args.k//128, 128)
            * (matrix_scale_act[:,0:args.m].t().view(args.m, args.k//128, 1)))
            .view(args.m, args.k) \
            @ (((matrix_B_int4.t().view(args.k//128, 128, partial_n_int4).int() \
                - matrix_zero[:,0:partial_n_int4].view(args.k//128, 1, partial_n_int4)).float()\
            * (matrix_scale_int4[:,0:partial_n_int4].view(args.k//128, 1, partial_n_int4))).view(args.k, partial_n_int4)))

    c_int8 = ((matrix_A.float().view(args.m, args.k//128, 128)
            * (matrix_scale_act[:,0:args.m].t().view(args.m, args.k//128, 1)))
            .view(args.m, args.k) \
            @ (((matrix_B_int8.t().view(args.k//128, 128, partial_n_int8)).float() \
            * (matrix_scale_int8[:,0:partial_n_int8].view(args.k//128, 1, partial_n_int8))).view(args.k, partial_n_int8)))

    c = torch.empty((args.m, args.n), dtype=torch.float32, device="cuda:0")
    c[:, matrix_indices_int4] = c_int4
    c[:, matrix_indices_int8] = c_int8

    try:
        torch.testing.assert_close(c_mixllm.half(),
                                   c.half(),
                                   rtol=3e-2,
                                   atol=2e-3)
        print(bcolors.OKBLUE + "Passed!" + bcolors.ENDC)
    except Exception as e:
        print(e)
        mismatched = (c_mixllm.half() -
                      c.half()).abs() > 2e-3 + 3e-2 * c.half().abs()
        if mismatched.sum() / mismatched.numel() > 1e-3:
            print(bcolors.FAIL + "Failed!" + bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + "Passed, within tolerance!" + bcolors.ENDC)
