# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import argparse
import mixllm.nn.modules.ops
from mixllm.nn.modules.utils import bcolors, time_logger, weight_quantize

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
    input = torch.randn(args.m, args.n, dtype=torch.float16, device="cuda:0")

    ##############################
    # run mixllm kernel
    ##############################
    input_quantized, act_scale_padded = mixllm.nn.modules.ops.quantize(input)
    torch.cuda.synchronize()

    ##############################
    # check correctness
    ##############################
    act_scale_padded_ref = torch.empty((args.n // 128, args.m + args.m % 2),
                                       dtype=torch.float16,
                                       device="cuda:0")
    act_scale = input.view(args.m, input.shape[1] // 128,
                           128).abs().amax(dim=-1) / 127
    input_tmp = input.view(args.m, input.shape[1] // 128, 128) / act_scale.view(
        act_scale.shape[0], act_scale.shape[1], 1)
    input_quantized_ref = input_tmp.round().to(torch.int8).view(args.m, args.n)
    act_scale = act_scale.t().contiguous()
    act_scale_padded_ref[:, :act_scale.shape[1]] = act_scale.half().cuda(
    ).contiguous()

    try:
        torch.testing.assert_close(input_quantized, input_quantized_ref)
    except Exception as e:
        print(e)
        mismatched = (input_quantized.half() - input_quantized_ref.half()
                     ).abs() > 2e-3 + 3e-2 * input_quantized_ref.half().abs()
        if mismatched.sum() / mismatched.numel() > 1e-2:
            print(bcolors.FAIL + "Failed!" + bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + "Quant Passed, within tolerance!" +
                  bcolors.ENDC)
    else:
        print(bcolors.OKBLUE + "Quant Passed!" + bcolors.ENDC)

    try:
        torch.testing.assert_close(act_scale_padded, act_scale_padded_ref)
    except Exception as e:
        print(e)
        mismatched = (act_scale_padded.half() - act_scale_padded_ref.half()
                     ).abs() > 2e-3 + 3e-2 * act_scale_padded_ref.half().abs()
        if mismatched.sum() / mismatched.numel() > 1e-3:
            print(bcolors.FAIL + "Failed!" + bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + "Scale Passed, within tolerance!" +
                  bcolors.ENDC)
    else:
        print(bcolors.OKBLUE + "Scale Passed!" + bcolors.ENDC)

    _, latency = time_logger(mixllm.nn.modules.ops.quantize, input)
    latency = latency * 1000
    print(bcolors.WARNING + "latency =", "%.2f" % latency, "us," + bcolors.ENDC)
