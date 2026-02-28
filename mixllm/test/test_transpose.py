# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import argparse
import torch
import mixllm.nn.modules.ops
from mixllm.nn.modules.utils import bcolors, time_logger

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=512)
parser.add_argument('-n', type=int, default=4096)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--warm_up', type=int, default=10)
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    ##############################
    # generate random input data
    ##############################
    input = torch.randn(args.n, args.m, dtype=torch.float16, device="cuda:0")

    ##############################
    # run and profile mixllm kernel
    ##############################
    output = mixllm.nn.modules.ops.transpose(input)
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(output, input.t())
    except Exception as e:
        print(e)
    else:
        print(bcolors.OKBLUE + "Passed!" + bcolors.ENDC)

    _, latency = time_logger(mixllm.nn.modules.ops.transpose, input)
    latency = latency * 1000
    print(bcolors.WARNING + "latency =", "%.2f" % latency, "us," + bcolors.ENDC)

    # cudagraph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(cudagraph):
    #     output = mixllm.nn.modules.ops.transpose(input)
