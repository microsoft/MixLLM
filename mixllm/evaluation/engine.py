# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
from typing import Union, List, Tuple
import logging

from mixllm.quantization.quantizer import (Quantizer, MixLLMConfig)
from mixllm.quantization.searcher import MixLLMSearcher
import mixllm.utils.modeling as modeling
from mixllm.utils.datautils import DataUtils

logger = logging.getLogger('mixllm')


def quantize_fake(pretrained_model_name_or_path: str, calib_input: torch.Tensor,
                  mixllm_config: MixLLMConfig, device: str) -> torch.nn.Module:
    import time
    '''
    It will quantize the model inplace. Only the linear layers in the
    decoder blocks will be quantized.
    '''

    start_time = time.time()
    mixllm_config = MixLLMSearcher.search_mix_config(
        pretrained_model_name_or_path, calib_input, mixllm_config, device)

    end_time = time.time()
    time_in_mins = (end_time - start_time) / 60
    logger.info(f"{round(time_in_mins, 2)} min for searching mix config.")

    model = modeling.get_model(pretrained_model_name_or_path,
                               torch_dtype=torch.float16)

    start_time = time.time()
    # TODO: support device map.
    Quantizer.quantize_model_fake(model, calib_input, mixllm_config)
    end_time = time.time()
    time_in_mins = (end_time - start_time) / 60
    logger.info(f"{round(time_in_mins, 2)} min for quantizing w/ mix config.")

    return model


def get_datasets(dataset_name: str,
                 pretrained_model_name_or_path: str,
                 nsamples: int = 128,
                 seed: int = 0,
                 device: str = 'cpu'):
    seqlen = modeling.get_seqlen(pretrained_model_name_or_path)
    trainloader, testenc = DataUtils.get_loaders(
        dataset_name,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        model=pretrained_model_name_or_path)
    # traindata = [inp.to(device) for inp, _ in trainloader]
    # traindata = torch.cat(traindata, dim=0)
    # traindata = DataUtils.trainloader_to_tensor(trainloader, device)
    testdata = testenc.input_ids.to(device)

    return trainloader, testdata,


def evaluate_ppl(model,
                 pretrained_model_name_or_path: str,
                 datasets: Union[List[str], List[Tuple[str, torch.Tensor]]],
                 device: str = 'cuda',
                 layer_swap: bool = False) -> dict:
    ppls = dict()
    seqlen = modeling.get_seqlen(pretrained_model_name_or_path)
    for dataset in datasets:
        if isinstance(dataset, tuple):
            name, input_ids = dataset
        else:
            _, testloader = DataUtils.get_loaders(
                dataset,
                nsamples=128,
                seed=0,
                seqlen=seqlen,
                model=pretrained_model_name_or_path)
            name = dataset
            input_ids = testloader.input_ids

        ppl = modeling.eval_ppl(model, input_ids, seqlen, f"{name}", device,
                                layer_swap)
        ppls[name] = ppl

    return ppls


def evaluate_downstream(model, downstream_tasks_map, device='cuda'):
    if not device == 'auto':
        orig_device = model.device
        print(f"Moving model to {device}")
        model.to(device)
    accuracy = modeling.eval_downstream(model, downstream_tasks_map)
    if not device == 'auto':
        model.to(orig_device)

    return accuracy
