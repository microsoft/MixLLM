# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import os
import gc
import logging
import torch

# TODO: rewrite the package that can be included.
from mixllm.quantization.quantizer import MixLLMConfig
import mixllm.evaluation.engine as engine
from mixllm.utils.datautils import DataUtils
from transformers import AutoTokenizer
from mixllm.nn.modules import utils


def __eval_base(pretrained_model_name_or_path, calib_datasets, bit_width_list,
                group_size: int, asym: bool, ppl_dataset_list: list,
                device: str):
    import mixllm.utils.modeling as modeling
    results = {}
    for bit_width in bit_width_list:
        if bit_width == 16:
            model = modeling.get_model(pretrained_model_name_or_path,
                                       torch_dtype=torch.float16,
                                       device_map=device)
        else:
            bit_percent = {bit_width: 100}
            weight_config = {
                bit_width: {
                    'group_size': group_size,
                    'asymmetric': asym,
                },
            }
            activation_config = None
            mixllm_config = MixLLMConfig(bit_percent=bit_percent,
                                         weight_config=weight_config,
                                         activation_config=activation_config)

            model = engine.quantize_fake(pretrained_model_name_or_path,
                                         calib_input=calib_datasets,
                                         mixllm_config=mixllm_config,
                                         device=device)

        res = {}

        if len(ppl_dataset_list) > 0:
            ppls = engine.evaluate_ppl(model,
                                       pretrained_model_name_or_path,
                                       ppl_dataset_list,
                                       device=device)
            ppls = {k: round(v, 6) for k, v in ppls.items()}
            res.update(ppls)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if bit_width == 16:
            key = 'fp16'
        else:
            key = f'rtn_w{bit_width}_g{group_size}_asym'
        results[key] = res
        print(f"{key}: {res}")

    return results


def __eval_mixllm(pretrained_model_name_or_path, calib_datasets, bit_percents,
                  weight_config_base: dict, activation_config_base: dict,
                  ppl_dataset_list: list, device: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    results = {}
    for bit_percent in bit_percents:
        logger = logging.getLogger('mixllm')
        logger.setLevel(logging.INFO)

        bit_percent = {bit: pct for bit, pct in bit_percent.items() if pct > 0}
        weight_config = {
            bit: conf
            for bit, conf in weight_config_base.items()
            if bit in bit_percent
        }
        activation_config = activation_config_base

        mixllm_config = MixLLMConfig(bit_percent=bit_percent,
                                     weight_config=weight_config,
                                     activation_config=activation_config)

        model_quant = engine.quantize_fake(pretrained_model_name_or_path,
                                           calib_input=calib_datasets,
                                           mixllm_config=mixllm_config,
                                           device=device)
        print(model_quant)

        res = {}

        if len(ppl_dataset_list) > 0:
            ppls = engine.evaluate_ppl(
                model_quant,
                pretrained_model_name_or_path,
                ppl_dataset_list,
                device='cuda:0'  # only supports single GPU now.
            )
            ppls = {k: round(v, 6) for k, v in ppls.items()}
            res.update(ppls)

        # save quant model for vllm inference
        ratio = bit_percent[8] / 100 if 8 in bit_percent else 0
        save_dir = "../quantized_model/" + pretrained_model_name_or_path + "/" + str(
            ratio)
        utils.save_for_vllm(model_quant, tokenizer, ratio, save_dir)

        del model_quant
        gc.collect()
        torch.cuda.empty_cache()

        config_4bit = weight_config_base[4]
        gptq_4bit = config_4bit['gptq']
        clip_4bit = config_4bit['clip_shrink']
        group_size = config_4bit['group_size']
        gptq_group_reorder = config_4bit['gptq_group_reorder']
        assert config_4bit['asymmetric'] is True

        config_8bit = weight_config_base[8]
        gptq_8bit = config_8bit['gptq']
        clip_8bit = config_8bit['clip_shrink']
        assert config_8bit['group_size'] == group_size
        assert config_8bit['gptq_group_reorder'] == gptq_group_reorder
        assert config_8bit['asymmetric'] is False

        percent_8bit = bit_percent.get(8, 0)
        act_bits = activation_config['bit_width']
        bits_info = f'w4p{percent_8bit}a{act_bits}'
        sym_info = f"asym4{config_4bit['asymmetric']}8{config_8bit['asymmetric']}"
        gptq_conf = f'gptq{gptq_4bit}{gptq_8bit}_reo{gptq_group_reorder}'
        clip_conf = f'clip{clip_4bit}{clip_8bit}'
        key = f'mixllm_{bits_info}_g{group_size}_{sym_info}_{gptq_conf}_{clip_conf}'
        results[key] = res
        print(f"{key}: {res}")

    return results


def __eval_model(pretrained_model_name_or_path):

    print(f"Start evaluating {pretrained_model_name_or_path}.")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tasks to evaluate.
    # ppl_tasks = ['wikitext2', 'c4']
    ppl_tasks = ['wikitext2']  # for quick test.

    # Baseline quantization methods to compare.
    baseline_rtn_list = [4, 16]  # weight only quantization in group-wise
    eval_mixllm = True

    if True:
        baseline_rtn_list = []  # Only for debugging.

    # Prepare datasets for PPL evaluation.
    nsamples = 128
    seed = 0
    calib_dataset_name = 'wikitext2'
    print(f"calib dataset: {calib_dataset_name}, nsamples: {nsamples}; "
          f"ppl tasks: {ppl_tasks}, baseline RTN: {baseline_rtn_list}, "
          f"eval MixLLM: {eval_mixllm}")
    calib_loader, _ = engine.get_datasets(calib_dataset_name,
                                          pretrained_model_name_or_path,
                                          nsamples,
                                          seed,
                                          device='cuda:0')
    calib_datasets, _ = DataUtils.trainloader_to_tensor(calib_loader,
                                                        device='cuda:0')
    ppl_dataset_list = []
    for dataset_name in ppl_tasks:
        _, eval_dataset = engine.get_datasets(dataset_name,
                                              pretrained_model_name_or_path,
                                              nsamples,
                                              seed,
                                              device='cuda:0')
        ppl_dataset_list.append((dataset_name, eval_dataset))

    group_size = 128
    asym_baselines = True
    if ('70B' in pretrained_model_name_or_path) or (
            '32B' in pretrained_model_name_or_path) or (
                '72B' in pretrained_model_name_or_path) or (
                    '8x7B' in pretrained_model_name_or_path):
        # The model is too large.
        device = 'auto'
    else:
        device = 'cuda'

    results = {}

    # float16 and basic RTN quantization evaluation.
    results_base = __eval_base(pretrained_model_name_or_path,
                               calib_datasets,
                               baseline_rtn_list,
                               group_size,
                               asym_baselines,
                               ppl_dataset_list,
                               device=device
                               # device='cuda:0'  # for parallel evaluation.
                              )
    results.update(results_base)

    if eval_mixllm:
        bit_percents = [{
            4: 100,
            8: 0,
        }, {
            4: 90,
            8: 10,
        }, {
            4: 0,
            8: 100,
        }]
        # bit_percents = [{8: 10, 4: 90}]  # for quick test
        # bit_percents = [{8: 0, 4: 100}]  # Only for debugging.

        gptq_4bit = True
        clip_4bit = True
        gptq_8bit = True
        clip_8bit = True
        gptq_group_reorder = True
        if ('70B' in pretrained_model_name_or_path) or (
                '32B' in pretrained_model_name_or_path) or (
                    '72B' in pretrained_model_name_or_path) or (
                        '8x7B' in pretrained_model_name_or_path):
            # The clip is too time consuming.
            clip_4bit = False
            clip_8bit = False
        weight_config = {
            4: {
                'group_size': group_size,
                'asymmetric': True,
                'gptq': gptq_4bit,
                'gptq_group_reorder': gptq_group_reorder,
                'clip_shrink': clip_4bit,
            },
            8: {
                'group_size': group_size,
                'asymmetric': False,
                'gptq': gptq_8bit,
                'gptq_group_reorder': gptq_group_reorder,
                'clip_shrink': clip_8bit,
            }
        }
        activation_config = {
            'bit_width': 8,
            'group_size': group_size,
            'asymmetric': False
        }

        logger = logging.getLogger('mixllm')
        logger.setLevel(logging.INFO)

        mixllm_results = __eval_mixllm(pretrained_model_name_or_path,
                                       calib_datasets,
                                       bit_percents,
                                       weight_config,
                                       activation_config,
                                       ppl_dataset_list,
                                       device=device)
        results.update(mixllm_results)

    return results


def run_eval():
    workloads = [
        # 'NousResearch/Llama-3.2-1B',
        # 'NousResearch/Meta-Llama-3.1-8B',
        # 'NousResearch/Meta-Llama-3.1-70B',
        # 'Qwen/Qwen2.5-0.5B',  # for quick test.
        'Qwen/Qwen2.5-1.5B',
        # 'Qwen/Qwen2.5-7B',
        # 'Qwen/Qwen2.5-32B',
        # 'mistralai/Mistral-7B-v0.3',
        # 'mistralai/Mixtral-8x7B-v0.1',    # Not supported by some baselines.
    ]

    results = {}
    for model_name in workloads:
        res = __eval_model(model_name)
        print(f"results of {model_name}: {res}")
        results[model_name] = res
    print(f"All evaluations are done. Results raw: {results}")

    # It evaluates each model on several quantization benchmarks for several
    # metrics.
    benchmarks = list(results[workloads[0]].keys())
    print(f"benchmarks: {benchmarks}")
    metrics = list(results[workloads[0]][benchmarks[0]].keys())
    print(f"metrics: {metrics}")
    table = ',' + ','.join(metrics)
    for workload in workloads:
        for benchmark in benchmarks:
            table += f'\n{workload}|{benchmark}'
            for metric in metrics:
                table += f',{results[workload][benchmark][metric]}'

    print(f"Table: {table}")


if __name__ == '__main__':
    run_eval()
