# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import argparse
from mixllm.nn.modules.linear import LinearMixLLM
from mixllm.nn.modules.utils import bcolors, time_logger

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=512)
parser.add_argument('-n', type=int, default=4096)
parser.add_argument('-k', type=int, default=4096)
parser.add_argument('-r', type=float, default=0.25)
args = parser.parse_args()


def test_linear_module():
    int4_start_idx = int(args.n * args.r)
    partial_n_int4 = args.n - int4_start_idx
    partial_n_int8 = int4_start_idx
    min_int8 = -127
    max_int8 = -1 * min_int8 + 1
    groupsize = 128
    num_groups = args.k // groupsize
    is_row_major = (partial_n_int8 == 0 or partial_n_int4 == 0)
    print(args)

    ##############################
    # generate random input data
    ##############################
    input = torch.randn((args.m, args.k), dtype=torch.float16, device="cuda:0")
    matrix_zero = torch.randint(0,
                                16, (num_groups, partial_n_int4),
                                dtype=torch.uint8,
                                device="cuda:0")
    matrix_scale_int8 = 0.1 * torch.randn(
        (num_groups, partial_n_int8), dtype=torch.float16, device="cuda:0")
    matrix_scale_int4 = 0.1 * torch.randn(
        (num_groups, partial_n_int4), dtype=torch.float16, device="cuda:0")
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

    lmixllm = LinearMixLLM(weight_int8=matrix_B_int8,
                           weight_int4=matrix_B_int4,
                           weight_scale_int8=matrix_scale_int8,
                           weight_scale_int4=matrix_scale_int4,
                           weight_zero_int4=matrix_zero,
                           indices_int8=matrix_indices_int8,
                           indices_int4=matrix_indices_int4)

    c_mixllm, matrix_A, matrix_scale_act = lmixllm(input, True)
    # import pdb; pdb.set_trace()

    ##############################
    # run and profile mixllm linear
    ##############################
    cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cudagraph):
        c_mixllm, _, _ = lmixllm(input, True)

    _, latency = time_logger(cudagraph.replay)
    print(bcolors.WARNING + "latency =", "%.2f" % latency, "ms,",
          "%.2f" % (args.m * args.n * args.k * 2 / latency / 1e9),
          "TFLOPs" + bcolors.ENDC)

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
    except Exception as e:
        print(e)
    else:
        print(bcolors.OKBLUE + "Passed!" + bcolors.ENDC)


if __name__ == "__main__":
    test_linear_module()
