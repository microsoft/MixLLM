# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
from mixllm.nn.modules.mixllm_config import MixLLMConfig
from huggingface_hub import save_torch_state_dict


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def time_logger(func, *pargs, **kwargs):
    n_warm_up = 10
    n_iter = 100
    import time
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("warmup")
    # warm up
    for i in range(n_warm_up):
        output = func(*pargs, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("run")
    t0 = time.time()

    for i in range(n_iter):
        output = func(*pargs, **kwargs)

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    t1 = time.time()
    latency = (t1 - t0) * 1000 / n_iter
    return (output, latency)


def weight_quantize(weight_fp16, ratio=1):
    assert (len(weight_fp16.shape) == 2)
    k = weight_fp16.shape[1]
    n = weight_fp16.shape[0]

    int4_start_idx = int(n * ratio)
    partial_n_int4 = n - int4_start_idx
    partial_n_int8 = int4_start_idx

    is_row_major = (partial_n_int8 == 0 or partial_n_int4 == 0)
    indices = torch.randperm(
        n, dtype=torch.int32,
        device="cuda:0") if not is_row_major else torch.arange(
            n, dtype=torch.int32, device="cuda:0")
    indices_int8 = indices[0:partial_n_int8]
    indices_int4 = indices[partial_n_int8:n]

    #######################
    # int8 part
    #######################
    weight_fp16_gathered_for_int8 = weight_fp16[indices_int8, :]
    weight_scale_int8 = weight_fp16_gathered_for_int8.view(
        partial_n_int8, k // 128, 128).abs().amax(dim=-1) / 127
    weight_int8 = weight_fp16_gathered_for_int8.view(partial_n_int8, k//128, 128) / \
                  weight_scale_int8.view(weight_scale_int8.shape[0], weight_scale_int8.shape[1], 1)
    weight_int8 = weight_int8.round().to(torch.int8).view(partial_n_int8, k)
    weight_scale_int8 = weight_scale_int8.t()

    #######################
    # int4 part
    #######################
    weight_fp16_gathered_for_int4 = weight_fp16[indices_int4, :]
    max_val = weight_fp16_gathered_for_int4.view(partial_n_int4, k // 128,
                                                 128).amax(dim=-1, keepdim=True)
    min_val = weight_fp16_gathered_for_int4.view(partial_n_int4, k // 128,
                                                 128).amin(dim=-1, keepdim=True)
    weight_scale_int4 = (max_val - min_val).clamp(min=1e-5) / 15
    weight_zero_int4 = (-torch.round(min_val / weight_scale_int4)).clamp_(0, 15)

    weight_int4 = weight_fp16_gathered_for_int4.view(partial_n_int4, k//128, 128) / \
                  weight_scale_int4.view(weight_scale_int4.shape[0], weight_scale_int4.shape[1], 1)
    weight_int4 = (weight_int4.round().to(torch.int8) + weight_zero_int4).clamp(
        0, 15).view(partial_n_int4, k)

    weight_scale_int4 = weight_scale_int4.view(partial_n_int4, k // 128).t()
    weight_zero_int4 = weight_zero_int4.view(partial_n_int4, k // 128).t()

    return weight_int8, weight_scale_int8, indices_int8, weight_int4, weight_scale_int4, weight_zero_int4, indices_int4


def save_for_vllm(model, tokenizer, ratio, save_dir):

    class EmptyModule(torch.nn.Module):

        def __init__(self):
            super(EmptyModule, self).__init__()

        def forward(self, x):
            return x

    quant_config = MixLLMConfig.from_dict({"ratio": ratio})

    # Save model and config files with empty state dict
    model.config.quantization_config = quant_config.to_transformers_dict()
    model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

    save_torch_state_dict(
        state_dict=model.state_dict(),
        save_directory=save_dir,
        max_shard_size="5GB",
        safe_serialization=True,
        force_contiguous=True,
        shared_tensors_to_discard=model._tied_weights_keys,
    )

    tokenizer.save_pretrained(save_dir)
