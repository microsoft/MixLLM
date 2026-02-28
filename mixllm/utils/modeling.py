# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import Union
from transformers import PreTrainedModel
from transformers import (LlamaForCausalLM, LlamaModel, MistralForCausalLM,
                          MistralModel, Qwen2ForCausalLM, Qwen2Model)
from transformers import AutoConfig
from tqdm import tqdm
# import functools
import gc
import logging

logger = logging.getLogger('mixllm')


def is_supported_model_causal_lm(model):
    return isinstance(model,
                      (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM))


def is_supported_model(model):
    return isinstance(model, (LlamaModel, MistralModel, Qwen2Model))


def supported_model_type():
    return (LlamaModel, MistralModel, Qwen2Model)


@torch.no_grad()
def get_input_embeddings(model):
    if is_supported_model_causal_lm(model):
        return model.get_input_embeddings()
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def get_output_embeddings(model):
    if is_supported_model_causal_lm(model):
        return model.get_output_embeddings()
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def get_rotary_embeddings(model):
    if hasattr(model.model, 'rotary_emb'):
        return model.model.rotary_emb
    else:
        return None


@torch.no_grad()
def get_norm(model):
    if hasattr(model.model, 'norm'):
        return model.model.norm
    else:
        return None


@torch.no_grad()
def get_seqlen(model: Union[torch.nn.Module, str], max_length: int = 2048):
    if isinstance(model, str):
        config = AutoConfig.from_pretrained(model)
    elif is_supported_model_causal_lm(model):
        config = model.config
    else:
        return max_length
    return min(config.max_position_embeddings, max_length)


@torch.no_grad()
def get_named_transformer_layers(model):
    if is_supported_model_causal_lm(model):
        layers = model.get_decoder().layers
    elif is_supported_model(model):
        layers = model.layers
    else:
        raise NotImplementedError(type(model))
    named_layers = []
    for name, module in model.named_modules():
        if module in layers:
            named_layers.append((name, module))
    return named_layers


@torch.no_grad()
def get_named_linears_in_transformer_layers(model,
                                            skip_moe_router: bool = True,
                                            max_num_experts: int = 64):
    named_layers = get_named_transformer_layers(model)
    named_linears = []
    for layer_name, layer in named_layers:
        for name, m in layer.named_modules(prefix=f"{layer_name}"):
            if not isinstance(m, nn.Linear):
                continue
            if skip_moe_router and (min(m.in_features, m.out_features)
                                    <= max_num_experts):
                continue
            named_linears.append((name, m))
    return named_linears


@torch.no_grad()
def get_transformer_layers(model):
    if is_supported_model_causal_lm(model):
        layers = model.get_decoder().layers
    else:
        raise NotImplementedError(type(model))
    return layers


def get_model(model_id, torch_dtype, device_map=None):

    def skip(*args, **kwargs):
        pass

    # Avoid the overhead of weight initialization.
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoModelForCausalLM

    # print(f"model_id: {model_id}, device_map: {device_map}")

    # if model_id == 'Qwen/Qwen2.5-72B':
    #     config = AutoConfig.from_pretrained(model_id)
    #     print(f"orig config: {config}")
    #     # config.rms_norm_eps = 1e-6
    #     print(f'new config: {config}')
    # else:
    #     config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # config=config,
        # attn_implementation="flash_attention_2"
    ).eval()

    # debug.dump_mem("after model loading")

    return model


def get_decoder(model):
    if is_supported_model_causal_lm(model):
        return model.get_decoder()
    else:
        raise NotImplementedError(type(model))


def get_debug_model(config: str = 'dense,tiny'):
    type, size = config.lower().split(',')
    if type == 'dense':
        from transformers import MistralConfig
        config = MistralConfig()

        if size == 'tiny':
            config.num_hidden_layers = 2
            config.hidden_size = 2
            config.intermediate_size = 4
            config.num_attention_heads = 2
            config.num_key_value_heads = 2
            config.vocab_size = 8
        elif size == 'small':
            config.num_hidden_layers = 2,
            config.hidden_size = 512,
            config.intermediate_size = 1024,
            config.num_attention_heads = 4,
            config.num_key_value_heads = 4,
            config.vocab_size = 1536
        elif size == 'medium':
            config.num_hidden_layers = 2
        elif size == 'full':
            pass
        else:
            raise ValueError(f"Invalid size: {size}")

        model = MistralModel(config)
    elif type == 'moe':
        raise NotImplementedError("MoE model is not implemented yet.")
        # from transformers import MixtralConfig
        # config = MixtralConfig()

        # if size == 'medium':
        #     config.num_hidden_layers = 2
        # else:
        #     raise ValueError(f"Invalid size: {size}")

        # model = MixtralModel(config)
    else:
        raise ValueError(f"Invalid type: {type}")

    return model


@torch.no_grad()
def get_named_leaf_modules(model):
    leaf_modules = []
    for name, module in model.named_modules():
        # print(f"name: {name}")
        if not list(module.children()):
            leaf_modules.append((name, module))
    # exit(0)
    return leaf_modules


@torch.no_grad()
def move_non_decoder_linear_to_device(model, device):
    transformer_linears = set([
        linear for _, linear in get_named_linears_in_transformer_layers(model)
    ])

    all_modules = set([module for _, module in get_named_leaf_modules(model)])
    target_modules = all_modules - transformer_linears
    for module in target_modules:
        module.to(device)


@torch.no_grad()
def eval_ppl(model: PreTrainedModel,
             dataset_eval: torch.Tensor,
             seqlen: int,
             info: str,
             device='cuda:0',
             layer_swap: bool = True):

    if True:
        print(f"device: {device}")

    if hasattr(model, 'hf_device_map'):
        if True:
            print(f"device map: {model.hf_device_map}")
        device_map = model.hf_device_map
        for name, m in model.named_modules():
            if name in device_map:
                m.to(device_map[name])
    elif not layer_swap:
        model.to(device)

    if dataset_eval.dim() == 1:
        dataset_eval = dataset_eval.unsqueeze(0)
    elif dataset_eval.dim() > 2:
        raise ValueError(f"Invalid input shape: {dataset_eval.shape}")
    effective_length = dataset_eval.size(1) // seqlen * seqlen
    dataset_eval = dataset_eval[:, :effective_length]

    if not device == 'auto':
        move_non_decoder_linear_to_device(model, device)
        dataset_eval.to(device)
    else:
        dataset_eval.to(next(model.parameters()).device)

    @torch.no_grad()
    def swap_in_pre_hook(module, args):
        # Swap the module to the GPU.
        module.cuda()

    @torch.no_grad()
    def swap_out_post_hook(module, args, output):
        # Swap the module back to the CPU.
        module.cpu()

    handles = []
    named_linears = get_named_linears_in_transformer_layers(model)
    for _, linear in named_linears:
        if layer_swap and (not device == 'auto'):
            pre_handle = linear.register_forward_pre_hook(swap_in_pre_hook)
            handles.append(pre_handle)
            handle = linear.register_forward_hook(swap_out_post_hook)
            handles.append(handle)

    input_ids = dataset_eval[:, :effective_length]
    # if not device == 'auto':
    #     input_ids = input_ids.to(device)
    input_ids = input_ids.view(-1, seqlen)
    effective_samples = input_ids.size(0)

    nlls = []
    # TODO: use a tiled version to avoid the frequent memory swapping.
    for idx, input in tqdm(enumerate(input_ids),
                           desc=f"evaluating [{info}]...",
                           total=effective_samples):
        with torch.no_grad():
            input = input.unsqueeze(0)
            loss = model(input, labels=input, use_cache=False).loss
            if loss.isinf() or loss.isnan():
                logger.warning(f"Invalid loss: {loss}, skip sample {idx}.")
                continue
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (effective_samples * seqlen))

    for handle in handles:
        handle.remove()
    gc.collect()

    return ppl.item()


@torch.no_grad()
def eval_downstream(model: PreTrainedModel, downstream_tasks_map: dict):
    '''
    Metrics to evaluate:
        wikitext, piqa, arc_easy, arc_challenge, boolq,
        xstorycloze_en, hellaswag, winogrande, lambada_standard, openbookqa
    To be included:
        copa
    '''

    import re
    import lm_eval
    import lm_eval.models
    # from lm_eval.models.vllm_causallms import VLLM

    # Only support huggerface format.
    lm = lm_eval.models.huggingface.HFLM(model)
    # lm = VLLM(model)
    # Can remove some of the tasks to speed up the evaluation: xstorycloze_en,
    # lambada_standard and openbookqa.

    all_results = {}
    for task_list, num_fewshot in downstream_tasks_map.items():
        tasks = task_list.split(',')
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            # limit=16,  # for debugging
            log_samples=False,
            verbosity="ERROR",
            batch_size='auto',
            cache_requests=True)
        results = results['results']
        if False:
            print(f"results: {results}")
        pattern = re.compile(r",none$")

        for task_name, result in results.items():
            result.pop('alias')
            result.pop(' ', None)
            if len(result) == 0:
                continue

            result_tmp = {}
            for metric_name, metric_value in result.items():
                metric_name_remove_none = re.sub(pattern, "", metric_name)
                result_tmp[metric_name_remove_none] = metric_value

            assert task_name not in all_results
            all_results[task_name] = result_tmp

        if False:
            for task_name in tasks:
                assert task_name in results
                task_result = results[task_name]

                task_result_tmp = {}
                for metric_name, metric_value in task_result.items():
                    metric_name_remove_none = re.sub(pattern, "", metric_name)
                    task_result_tmp[metric_name_remove_none] = metric_value

                if False:
                    print(f"{task_name} result: {task_result_tmp}")

                assert task_name not in all_results
                all_results[task_name] = task_result_tmp

    results_simple = {}
    for task_name, task_result in all_results.items():
        # task_result = all_results[task_name]
        if 'acc_norm' in task_result:
            accuracy = task_result['acc_norm'] * 100
        elif 'acc' in task_result:
            accuracy = task_result['acc'] * 100
        elif 'exact_match' in task_result:
            accuracy = task_result['exact_match'] * 100
        elif ('inst_level_strict_acc'
              in task_result) and ('prompt_level_strict_acc' in task_result):
            accuracy = (task_result['inst_level_strict_acc'] +
                        task_result['prompt_level_strict_acc']) / 2 * 100
        else:
            raise NotImplementedError(
                f'{task_result} for {task_name}, no supported metric found.')
        accuracy = round(accuracy, 2)
        results_simple[task_name] = accuracy
    # print(f"result simple: {results_simple}")

    return results_simple


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)
