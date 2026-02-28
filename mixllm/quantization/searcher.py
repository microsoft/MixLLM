# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import gc
import tqdm
import copy
import os
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Tuple, Set, Union
import logging

from mixllm.quantization.quantizer import (QuantConfig, MixLLMConfig, Quantizer)
from ..utils import modeling, debug

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = logging.getLogger('mixllm')


class MixLLMSearcher:
    __DEBUG = True

    @classmethod
    @torch.no_grad()
    def __sort_channels(cls, channel_loss_dict):
        # [(loss, linear_name, channel_idx), ...]
        all_channels: List[Tuple[float, str, int]] = []
        for name, loss in channel_loss_dict.items():
            channels = [
                (l, name, i) for i, l in enumerate(loss.cpu().detach().numpy())
            ]
            all_channels.extend(channels)
        # Channels with larger loss are placed in the front.
        sorted_channels = sorted(all_channels, key=lambda x: x[0], reverse=True)
        return sorted_channels

    @classmethod
    @torch.no_grad()
    def __loss_score_est(cls, model, input_ids, quant_fn, channel_updated_map):

        # TODO: use module.input = input, rather than using an external dict.
        module_input_dict = {}
        channel_loss_score_dict = {}
        # TODO: use setdefault for dict operation.

        named_linears = modeling.get_named_linears_in_transformer_layers(model)
        for name, _ in named_linears:
            # Only the unupdated channels will be 0, otherwise negative.
            channel_loss_score_dict[name] = channel_updated_map[name].to(
                torch.float).neg()

        def fwd_swapin_fim_pre_hook(module, args):
            assert isinstance(module, torch.nn.Linear)
            module_input_dict[module] = args[0]

        def bwd_swapin_fim_pre_hook(module, grad_output):
            assert isinstance(module, torch.nn.Linear)
            input = module_input_dict[module]
            # output = module_output_dict[module][0]
            # input.cuda()
            # output.cuda()

            assert len(grad_output) == 1
            if False:
                print(f"name: {module.name}, input: {input.shape}")
            assert (input.shape[0] == 1 and input.dim() == 3) or (input.dim()
                                                                  == 2)
            input = input.reshape(-1, input.shape[-1])
            grad_output = grad_output[0].reshape(-1, grad_output[0].shape[-1])
            grad_wrt_weight = torch.einsum('so,si->oi', grad_output.float(),
                                           input.float())
            name = module.name
            '''
            The calculation for each channel is:
            emp_fim = g * g^T
            channel_loss = 0.5 * w * emp_fim * w^T
                        = 0.5 * w * g * g^T * w^T
                        = 0.5 * (w * g) * (w * g)^T
                        = 0.5 * (w * g)**2
            Note (w * g) is a scalar for a channel.
            '''

            ignore_first_order = (os.environ.get('ignore_first_order',
                                                 'False') == 'True')
            use_diagnal = (os.environ.get('use_diagnal', 'False') == 'True')

            # ignore_first_order = False
            # use_diagnal = False

            quantized = quant_fn(module.weight)
            # delta = w - w_0, where w_0 is the point of Taylor Expansion.
            delta = (quantized - module.weight).float()
            del quantized
            if use_diagnal:
                grad_delta_mul = torch.mul(grad_wrt_weight, delta)
                scores = (grad_delta_mul.pow(2) * 0.5)
                if not ignore_first_order:
                    # first order loss is grad .dot delta_w
                    scores = scores + grad_delta_mul
                    scores = scores.abs()
                scores = scores.sum(dim=1)
            else:
                grad_delta_mul = torch.einsum('oi,oi->o', grad_wrt_weight,
                                              delta)
                scores = grad_delta_mul.pow(2) * 0.5
                if not ignore_first_order:
                    # first order loss is grad .dot delta_w
                    scores = scores + grad_delta_mul
                    scores = scores.abs()

            # TODO: double check if this method is correct.
            # Only update the unupdated channels.
            channel_loss_score_dict[name] += torch.logical_not(
                channel_updated_map[name]).to(scores.dtype) * scores.to(
                    channel_updated_map[name].device)

        fwd_handles = []
        bwd_handles = []
        for (name, module) in named_linears:
            module.name = name
            fwd_fim_handle = module.register_forward_pre_hook(
                fwd_swapin_fim_pre_hook)
            fwd_handles.append(fwd_fim_handle)
            bwd_fim_handle = module.register_full_backward_pre_hook(
                bwd_swapin_fim_pre_hook)
            bwd_handles.append(bwd_fim_handle)

        batch = input_ids.shape[0]
        embedding_fn = model.get_input_embeddings()
        module_input_dict.clear()
        model.zero_grad()
        self_target = False
        for inp in tqdm.tqdm(input_ids, desc='Channel loss est.', total=batch):
            inp = inp.unsqueeze(0).cuda()
            input_embed = embedding_fn(inp)
            if self_target:
                print(f"Using self target.")
                with torch.no_grad():
                    logits = model(inputs_embeds=input_embed,
                                   use_cache=False).logits
                    label = logits.argmax(dim=-1)
                    module_input_dict.clear()
            else:
                label = inp

            input_embed.requires_grad = True
            with torch.enable_grad():
                # debug.dump_mem("before model execution")
                # TODO: use torch.compile to optimize the model.
                loss = model(inputs_embeds=input_embed,
                             labels=label,
                             use_cache=False).loss
            # debug.dump_mem("after model execution")
            torch.autograd.grad(loss,
                                input_embed,
                                retain_graph=False,
                                create_graph=False)
            # debug.dump_mem("after grad")

            del input_embed
            module_input_dict.clear()
            model.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()

        for fwd_handle, bwd_handle in zip(fwd_handles, bwd_handles):
            fwd_handle.remove()
            bwd_handle.remove()

        return channel_loss_score_dict

    @classmethod
    @torch.no_grad()
    def search_mix_config(cls, pretrained_model_name_or_path, calib_input,
                          mixllm_config: MixLLMConfig,
                          device: str) -> MixLLMConfig:
        print(f"Base config. bit_percent: {mixllm_config.bit_percent}, "
              f"weight config {mixllm_config.weight_config}, "
              f"activation config {mixllm_config.activation_config}.")

        bit_percent = mixllm_config.bit_percent
        weight_config = mixllm_config.weight_config
        assert sum(bit_percent.values()) == 100
        ordered_bit_widths = list(bit_percent.keys())

        print(f"name: {pretrained_model_name_or_path}")
        if len(ordered_bit_widths) > 1:
            logger.info('Will use all the GPUs available for precision search.')
            model = modeling.get_model(pretrained_model_name_or_path,
                                       torch_dtype=torch.float16,
                                       device_map=device)
        else:
            # For the single bit-width case, we do not need the multiple GPUs.
            model = modeling.get_model(pretrained_model_name_or_path,
                                       torch_dtype=torch.float16)
        # print(f"device_map: {model.hf_device_map}")

        # if len(ordered_bit_widths) == 1:
        #     return mixllm_config

        ordered_bit_widths.sort(reverse=True)
        for bit_width in ordered_bit_widths:
            assert bit_width in weight_config

        # Move module onto GPU except the transformer linears.
        # modeling.move_non_decoder_linear_to_device(model, 'cuda')
        # debug.dump_mem("before perform searching")

        channel_updated_map = {}
        named_linears = OrderedDict(
            modeling.get_named_linears_in_transformer_layers(model))
        first_device = next(model.parameters()).device
        for name, linear in named_linears.items():
            out_features = linear.weight.size(0)
            channel_updated_map[name] = torch.zeros(out_features,
                                                    dtype=torch.int32,
                                                    device=first_device)
            config_map = {}
            # TODO: if there is only a single bit_width, skip the indices.
            for bit_width in ordered_bit_widths:
                basic_config = weight_config[bit_width]
                group_size = basic_config['group_size']
                asymmetric = basic_config['asymmetric']
                indices = list(range(0, out_features)
                              ) if bit_width == ordered_bit_widths[-1] else []
                config_map[bit_width] = {
                    'group_size': group_size,
                    'asymmetric': asymmetric,
                    'indices': indices
                }

            mixllm_config.linear_config_map[name] = QuantConfig(config_map)

        for curr_large_bit_idx, next_small_bit_width in enumerate(
                ordered_bit_widths[1:]):
            curr_large_bit_width = ordered_bit_widths[curr_large_bit_idx]
            large_bit_percent = bit_percent[curr_large_bit_width]
            print(f"curr large: {curr_large_bit_width}, "
                  f"next small: {next_small_bit_width}, "
                  f"large bit percent: {large_bit_percent}")

            # Try to enable the gptq and clip shrink for the quantization here.
            quant_fn_small_bit = partial(
                Quantizer.quantize_linear_weight,
                activation=None,
                config=QuantConfig({
                    next_small_bit_width: {
                        'group_size':
                            weight_config[next_small_bit_width]['group_size'],
                        'asymmetric':
                            weight_config[next_small_bit_width]['asymmetric']
                    }
                }))

            calib_input = calib_input.to(first_device)
            channel_loss_scores = cls.__loss_score_est(model, calib_input,
                                                       quant_fn_small_bit,
                                                       channel_updated_map)
            if cls.__DEBUG:
                for _, loss in channel_loss_scores.items():
                    assert loss.isnan().sum() == 0
            # The updated channels's loss value will be negative.
            sorted_channels = cls.__sort_channels(channel_loss_scores)
            # print(
            #     f"sorted: [{sorted_channels[0]}, ..., {sorted_channels[-1]}]")

            # Collect the new large-bit channels.
            large_bit_update_map: Dict[str, Set[int]] = {}
            n_channels = len(sorted_channels) * large_bit_percent // 100
            top_channels = sorted_channels[:n_channels]
            logger.info(
                f"top loss: [{top_channels[0]}, ..., {top_channels[-1]}]")
            # print(f"top loss: [{top_channels[0]}, ..., {top_channels[-1]}]")
            # Make sure the loss of top n_channels in sort_channels are valid.
            assert all([loss >= -1 for loss, _, _ in top_channels])
            for _, name, channel_idx in top_channels:
                if name not in large_bit_update_map:
                    large_bit_update_map[name] = set()
                large_bit_update_map[name].add(channel_idx)

            # Update small-bit channels, log the updated channels, and quantize
            # the newly added large-bit channels.
            for name, linear in named_linears.items():
                if not name in large_bit_update_map:
                    continue
                large_bit_updated_set = large_bit_update_map[name]
                new_large_bit_channels = list(large_bit_updated_set)
                new_large_bit_channels.sort()

                linear_config = mixllm_config.linear_config_map[name]
                large_bit_channels = linear_config.bit_config_map[
                    curr_large_bit_width]['indices']
                large_bit_channels.extend(new_large_bit_channels)
                linear_config.bit_config_map[curr_large_bit_width][
                    'indices'] = large_bit_channels

                # Exclude the updated large-bit channels from all the small-bit
                for smaller_bit in ordered_bit_widths[curr_large_bit_idx + 1:]:
                    small_bit_channels = linear_config.bit_config_map[
                        smaller_bit]['indices']
                    small_bit_channels = list(
                        set(small_bit_channels) - large_bit_updated_set)
                    linear_config.bit_config_map[smaller_bit][
                        'indices'] = small_bit_channels

                # Do not update the weight of large-bit channels in the current
                # implementation. May do it in the future.

                mixllm_config.linear_config_map[name] = linear_config
                channel_updated_map[name][new_large_bit_channels] = 1

        del model

        # Sort the indices, and update other configurations.
        for name, linear in named_linears.items():
            linear_config = mixllm_config.linear_config_map[name]
            for bit_width in linear_config.bit_config_map.keys():
                linear_config.bit_config_map[bit_width]['indices'].sort()
                basic_config = weight_config[bit_width]
                for key, val in basic_config.items():
                    if key not in linear_config.bit_config_map[bit_width]:
                        linear_config.bit_config_map[bit_width][key] = val
            linear_config.verify(linear.weight.shape)
            mixllm_config.linear_config_map[name] = linear_config

        return mixllm_config
