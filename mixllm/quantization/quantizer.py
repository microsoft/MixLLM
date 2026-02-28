# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
# from collections import OrderedDict
import copy
from tqdm import tqdm
import gc
from typing import Dict, Union, Tuple, Optional, List
from itertools import chain
from functools import partial
import math
import logging
from mixllm.nn.modules.linear_for_vllm import LinearMixLLM4vLLM

from ..utils import modeling, debug
from ..utils.debug import TimeUtils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

logger = logging.getLogger('mixllm')


class QuantConfig:
    """
    Quantization configuration for Linear layers.
    """

    def __init__(self, bit_config_map: Dict[int, Dict[str, Union[int, bool,
                                                                 List[int]]]]):
        """
        {
            bit_width: {
                'group_size': int, 
                'asymmetric': bool, 
                'gptq': bool,         # optional
                'clip_shrink': bool,  # optional
                'indices': List[int]  # optional for single bit-width.
            }
        }
        """
        self.bit_config_map = bit_config_map

    def verify(self, weight_shape: Tuple[int, int]):
        for bit_width, config in self.bit_config_map.items():
            assert bit_width > 0
            assert 'group_size' in config
            assert 'asymmetric' in config
        if len(self.bit_config_map) > 1:
            # If there are multiple bit-widths, at least one of them should have
            # indices.
            at_least_one_indices = False
            for _, config in self.bit_config_map.items():
                if 'indices' in config:
                    at_least_one_indices = True
                    break
            assert at_least_one_indices

            # Check if all indices are covered.
            all_indices = set()
            for _, config in self.bit_config_map.items():
                if 'indices' in config:
                    all_indices.update(config['indices'])
            assert all_indices == set(range(weight_shape[0]))

    def get_uniform_bit(self, weight_shape: Tuple[int, int]):
        # TODO: rewrite the related logic.
        if len(self.bit_config_map) == 1:
            return next(iter(self.bit_config_map.keys()))
        for bit_width, config in self.bit_config_map.items():
            if 'indices' not in config:
                continue
            if len(config['indices']) == weight_shape[0]:
                return bit_width
        return -1


# TODO: use a more general definition for the mixLLM configuration.
class MixLLMConfig:

    def __init__(self, bit_percent: Dict[int, int],
                 weight_config: Dict[int, Dict[str, Union[int, bool]]],
                 activation_config: Dict[str, Union[int, bool]]) -> None:
        # The general configuration.
        self.bit_percent = {
            bit: pct for bit, pct in bit_percent.items() if pct > 0
        }
        self.weight_config = {
            bit: conf
            for bit, conf in weight_config.items()
            if bit in self.bit_percent
        }
        self.activation_config = activation_config

        # The configuration of each linear layer.
        self.linear_config_map: Dict[str, QuantConfig] = dict()

        self.__verify()

    def __verify(self):
        # weight config
        assert self.bit_percent and (sum(self.bit_percent.values()) == 100)
        assert self.weight_config and (len(self.weight_config) > 0)
        for bit_width, percent in self.bit_percent.items():
            assert bit_width > 0
            assert percent > 0
            assert bit_width in self.weight_config
            assert 'group_size' in self.weight_config[bit_width]
            assert 'asymmetric' in self.weight_config[bit_width]

        # activation config
        if self.activation_config:
            act_bit_width = self.activation_config.get('bit_width', -1)
            # assert (act_bit_width == 8) or (act_bit_width == 16)
            act_group_size = self.activation_config.get('group_size', -1)
            # assert act_group_size > 0
            act_asymmetric = self.activation_config.get('asymmetric', False)
            assert act_asymmetric == False

    def save(self, path: str):
        torch.save(self.linear_config_map, path)

    def load(self, path: str):
        self.linear_config_map = torch.load(path)

    def __str__(self):
        return f"{self.linear_config_map}"

    def __repr__(self):
        return self.__str__()


class _EarlyExitException(Exception):
    pass


@torch.no_grad()
def _exit_fwd_pre_hook(module, input):
    raise _EarlyExitException


class QuantizedLinearLayer(nn.Linear):

    def __init__(self, activation_config=None, **kwargs):
        super(QuantizedLinearLayer, self).__init__(**kwargs)
        # TODO: add weight quantization here.
        self.activation_config = activation_config
        if self.activation_config:
            bit_width = self.activation_config['bit_width']
            group_size = self.activation_config.get('group_size', 128)
            asymmetric = self.activation_config.get('asymmetric', False)
            self.quant_fn = partial(Quantizer.quantize_activation,
                                    bit_width=bit_width,
                                    group_size=group_size,
                                    asymmetric=asymmetric,
                                    fake=True)
            self.activation_quant_bits = 8

    def forward(self, input):
        if self.activation_config:
            input = self.quant_fn(input)
        if self.bias is not None:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight)


class Quantizer(torch.autograd.Function):
    __DEBUG = True

    @classmethod
    def __get_quantize_param_raw(cls, input: torch.Tensor, bit_width: int,
                                 group_size: int, asymmetric: bool):
        '''
        Quantize along the row direction, which is the out_channels direction
        for the weight tensor and the token direction for the activation tensor
        by default.
        '''

        shape = input.shape
        device = input.device
        if input.dim() == 3:  # activation
            input = input.reshape(-1, shape[-1])
        n_rows, hidden_size = input.shape
        if group_size > 0:
            assert hidden_size % group_size == 0
        else:
            group_size = hidden_size
        input = input.reshape(-1, group_size)

        if asymmetric:
            min_val, max_val = input.aminmax(dim=1, keepdim=True)
            min_val = min_val.reshape(n_rows, -1)  # not neessary?
            max_val = max_val.reshape(n_rows, -1)
            min_int = 0
            max_int = 2**bit_width - 1
        else:
            min_val = 0
            max_val = input.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_val = max_val.reshape(n_rows, -1)
            min_int = -(2**(bit_width - 1))
            max_int = 2**(bit_width - 1) - 1

        return min_val, max_val, min_int, max_int

    @classmethod
    @torch.no_grad()
    def __shrinked_scales_zeros(cls, min_val, max_val, min_int, max_int, ratio,
                                asymmetric):
        min_val, max_val = min_val * ratio, max_val * ratio
        if asymmetric:
            # TODO: test removing the clamp.
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            scales = max_val / max_int
            zeros = 0
        return scales, zeros

    @classmethod
    @torch.no_grad()
    def __quantize_with_param(cls,
                              input: torch.Tensor,
                              bit_width: int,
                              group_size: int,
                              asymmetric: bool,
                              scales: torch.Tensor,
                              zeros: Union[torch.Tensor, int],
                              min_int: int,
                              max_int: int,
                              fake: bool = True):
        shape = input.shape
        input = input.reshape(-1, shape[-1])
        out_features, in_features = input.shape
        if group_size > 0:
            assert in_features % group_size == 0
        else:
            group_size = in_features
        num_groups_per_channel = in_features // group_size
        group_number = out_features * num_groups_per_channel
        input = input.reshape(group_number, group_size)
        scales = scales.reshape(group_number, 1)
        zeros = zeros.reshape(group_number, 1) if isinstance(
            zeros, torch.Tensor) else zeros
        quantized = torch.clamp(
            torch.round(input / scales) + zeros, min_int, max_int)
        if cls.__DEBUG:
            assert torch.isnan(quantized).sum() == 0
            assert torch.isinf(quantized).sum() == 0

        if fake:
            dequantized = (quantized - zeros) * scales
            return dequantized.reshape(shape)
        else:
            if bit_width == 8:
                dtype = torch.uint8 if asymmetric else torch.int8
            elif bit_width == 4:
                # To support the 4-bit, maybe by packing.
                dtype = torch.uint4 if asymmetric else torch.int4
            else:
                raise ValueError("Unsupported bit-width.")
            quantized = quantized.to(dtype).reshape(shape)
            shape[-1] = num_groups_per_channel
            scales = scales.reshape(shape)
            zeros = zeros.to(dtype).reshape(shape) if asymmetric else zeros
            return quantized, scales, zeros

    @classmethod
    @torch.compile
    @torch.no_grad()
    def __mse_with_delta_weight(cls, input, delta_weight, dim=None):
        mse = ((torch.einsum('ti,oi->to', input,
                             delta_weight)**2).mean(dim=dim))
        return mse

    @classmethod
    @torch.no_grad()
    def __search_clip_param_groupwise(cls, weight, activation, bit_width,
                                      group_size, asymmetric):
        out_features, in_features = weight.shape
        dtype = weight.dtype
        device = weight.device
        if group_size > 0:
            assert in_features % group_size == 0
        else:
            group_size = in_features

        min_val, max_val, min_int, max_int = cls.__get_quantize_param_raw(
            weight, bit_width, group_size, asymmetric)
        num_groups_per_channel = in_features // group_size
        group_number = out_features * num_groups_per_channel
        min_val = min_val.reshape(group_number, 1) if isinstance(
            min_val, torch.Tensor) else min_val
        max_val = max_val.reshape(group_number, 1)

        group_min_mse = torch.full((num_groups_per_channel, out_features),
                                   float('inf'),
                                   dtype=dtype,
                                   device=device)
        group_best_pct = torch.full((num_groups_per_channel, out_features),
                                    100,
                                    dtype=torch.int,
                                    device=device)
        # input = activation.reshape(-1, activation.shape[-1])
        # [s, n_g, g] -> [n_g, s, g]
        input_grouped = activation.reshape(-1, num_groups_per_channel,
                                           group_size).permute(1, 0,
                                                               2).contiguous()
        # [o, n_g, g] -> [n_g, o, g]
        weight_grouped = weight.reshape(out_features, num_groups_per_channel,
                                        group_size).permute(1, 0,
                                                            2).contiguous()

        # use_compile = True
        step = 2
        pct_grid = range(80, 101, step)
        for pct in pct_grid:
            ratio = pct / 100.0
            scales, zeros = cls.__shrinked_scales_zeros(min_val, max_val,
                                                        min_int, max_int, ratio,
                                                        asymmetric)
            quantized = cls.__quantize_with_param(weight,
                                                  bit_width,
                                                  group_size,
                                                  asymmetric,
                                                  scales,
                                                  zeros,
                                                  min_int,
                                                  max_int,
                                                  fake=True)
            if cls.__DEBUG:
                assert quantized.isnan().sum() == 0
                assert quantized.isinf().sum() == 0

            quantized = weight_grouped - quantized.reshape(
                out_features, num_groups_per_channel, group_size).permute(
                    1, 0, 2).contiguous()
            # # [out_features, num_groups_per_channel]
            # mse = (torch.einsum('tng,ong->ton', input_grouped,
            #                     quantized)**2).mean(dim=0)
            for gid, (input_g,
                      delta_g) in enumerate(zip(input_grouped, quantized)):
                if False:
                    debug.dump_mem("search_clip_param_groupwise before mse")
                    print(f"input_g GB: {debug.tensor_size_gigabytes(input_g)}")
                    print(f"delta_g GB: {debug.tensor_size_gigabytes(delta_g)}")
                # TODO: OOM for llama 70B model.
                # mse = ((torch.einsum('tg,og->to', input_g,
                #                      delta_g)**2).mean(dim=0))
                mse = cls.__mse_with_delta_weight(input_g, delta_g, dim=0)
                if False:
                    debug.dump_mem("search_clip_param_groupwise after mse")
                    print(f"mse GB: {debug.tensor_size_gigabytes(mse)}")
                update_mask = mse < group_min_mse[gid]
                group_min_mse[gid][update_mask] = mse[update_mask]
                group_best_pct[gid][update_mask] = pct
            del quantized
            if cls.__DEBUG:
                assert group_min_mse.isnan().sum() == 0
        del group_min_mse
        del input_grouped
        del weight_grouped

        # quantize according to the best pct.
        ratio = (group_best_pct / 100.0).to(dtype).t().contiguous().reshape(
            group_number, -1)
        scales, zeros = cls.__shrinked_scales_zeros(min_val, max_val, min_int,
                                                    max_int, ratio, asymmetric)
        scales = scales.reshape(out_features, num_groups_per_channel)
        zeros = zeros.reshape(out_features,
                              num_groups_per_channel) if isinstance(
                                  zeros, torch.Tensor) else zeros

        enable_fallback = True
        if enable_fallback:
            scales_base, zeros_base, _, _ = cls.__get_quant_param(
                weight, bit_width, group_size, asymmetric)
            quant_base = cls.__quantize_with_param(weight,
                                                   bit_width,
                                                   group_size,
                                                   asymmetric,
                                                   scales_base,
                                                   zeros_base,
                                                   min_int,
                                                   max_int,
                                                   fake=True)
            assert quant_base.isnan().sum() == 0
            assert quant_base.isinf().sum() == 0
            # mse_base = (torch.einsum('ti,oi->to',
            #                          activation.reshape(-1, in_features),
            #                          quant_base - weight)**2).mean()
            mse_base = cls.__mse_with_delta_weight(
                activation.reshape(-1, in_features), quant_base - weight)
            del quant_base
            quant_new = cls.__quantize_with_param(weight,
                                                  bit_width,
                                                  group_size,
                                                  asymmetric,
                                                  scales,
                                                  zeros,
                                                  min_int,
                                                  max_int,
                                                  fake=True)
            assert quant_new.isnan().sum() == 0
            assert quant_new.isinf().sum() == 0
            # mse_new = (torch.einsum('ti,oi->to',
            #                         activation.reshape(-1, in_features),
            #                         quant_new - weight)**2).mean()
            mse_new = cls.__mse_with_delta_weight(
                activation.reshape(-1, in_features), quant_new - weight)
            del quant_new
            # assert mse_new <= mse_base, f"{mse_new} not less than {mse_base}"

            if mse_new > mse_base:
                # Note the per-group smaller mse does not guarantee the global
                # smaller mse, as the per-group mse is calculated based on the
                # partial value on K dimension.
                scales = scales_base
                zeros = zeros_base
                logger.info(f"clip_shrink fallback for a tensor.")

        return scales, zeros, min_int, max_int

    @classmethod
    def __get_quant_param(cls, input: torch.Tensor, bit_width: int,
                          group_size: int, asymmetric: bool):
        '''
        Quantize along the row direction, which is the out_channels direction
        for the weight tensor and the token direction for the activation tensor
        by default.
        '''
        min_val, max_val, min_int, max_int = cls.__get_quantize_param_raw(
            input, bit_width, group_size, asymmetric)

        if asymmetric:
            # TODO: test removing the clamp.
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            scales = max_val / max_int
            zeros = 0

        return scales, zeros, min_int, max_int

    @classmethod
    @torch.no_grad()
    def __quantize_weight_rtn(cls,
                              weight: torch.Tensor,
                              activation: torch.Tensor,
                              bit_width: int,
                              group_size: int,
                              asymmetric: bool,
                              enable_clip_shrink: bool = False,
                              fake: bool = True):
        '''
        Quantize along the row direction, which is the out_channels direction
        for the weight tensor and the token direction for the activation tensor
        by default.
        '''
        if enable_clip_shrink:
            assert activation is not None
            scales, zeros, min_int, max_int = cls.__search_clip_param_groupwise(
                weight, activation, bit_width, group_size, asymmetric)
        else:
            scales, zeros, min_int, max_int = cls.__get_quant_param(
                weight, bit_width, group_size, asymmetric)

        return cls.__quantize_with_param(weight, bit_width, group_size,
                                         asymmetric, scales, zeros, min_int,
                                         max_int, fake)

    @classmethod
    @torch.compile
    def __hessinv(cls, H: torch.Tensor, percdamp: float):
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.size(0), device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H.to(torch.float32)
        return Hinv

    @classmethod
    @torch.no_grad()
    def __quantize_weight_gptq(
            cls,
            weight: torch.Tensor,
            activation: torch.Tensor,
            bit_width: int,
            group_size: int,
            asymmetric: bool,
            group_reorder: bool,
            #    reorder: bool,
            #    jit_group_param: bool,
            enable_clip_shrink: bool,
            percdamp: float = 0.01,
            fake: bool = True):
        '''
        Partial of the code is based on https://github.com/IST-DASLab/gptq.
        '''

        device = weight.device
        dtype = weight.dtype
        out_features, in_features = weight.shape
        # Transpose the weight to speedup the GPTQ execution (higher memory
        # efficiency).
        weight_update_t = weight.clone().detach().t().contiguous().to(
            torch.float32)
        if group_size > 0:
            assert in_features % group_size == 0
        else:
            group_size = in_features
        block_size = group_size

        hess_dtype = torch.float64
        assert activation.dim() == 3, "Only support 3D activation tensor."
        assert activation.size(-1) == in_features
        batch = activation.size(0)
        H = torch.zeros(in_features,
                        in_features,
                        dtype=hess_dtype,
                        device=device)
        for sample in activation:
            # Run different samples separately to save memory.
            tmp = math.sqrt(2 / batch) * sample.to(hess_dtype)
            H += torch.einsum('si,sj->ij', tmp, tmp)
            del tmp
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        weight_update_t[dead, :] = 0
        del dead

        num_groups_per_channel = in_features // group_size

        # if reorder:
        #     perm = torch.argsort(torch.diag(H), descending=True)
        #     weight_update_t = weight_update_t[perm, :]
        #     H = H[perm][:, perm]
        #     perm_inv = torch.argsort(perm)

        if group_reorder:
            grouped_diag_sum = torch.diag(H).reshape(-1, group_size).sum(1)
            group_perm = torch.argsort(grouped_diag_sum, descending=True)
            if cls.__DEBUG:
                assert (group_perm.dim() == 1) and (group_perm.numel()
                                                    == num_groups_per_channel)
            perm_base = (group_perm * group_size).expand(
                group_size, num_groups_per_channel).t().contiguous()
            offsets = torch.tensor(
                list(range(group_size)),
                device=weight.device).repeat(num_groups_per_channel).reshape(
                    num_groups_per_channel, group_size)
            perm = (perm_base + offsets).reshape(-1)
            del perm_base, offsets

            weight_update_t = weight_update_t[perm, :]
            H = H[perm][:, perm]

            perm_inv = torch.argsort(perm)
            group_perm_inv = torch.argsort(group_perm)

        use_compile = True
        if use_compile:
            Hinv = cls.__hessinv(H, percdamp)
        else:
            damp = torch.tensor(percdamp, dtype=hess_dtype,
                                device=device) * torch.mean(torch.diag(H))
            diag = torch.arange(in_features, device=device)
            H[diag, diag] += damp
            del damp, diag
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H.to(weight_update_t.dtype)
        del H

        if asymmetric:
            min_int = 0
            max_int = 2**bit_width - 1
        else:
            min_int = -(2**(bit_width - 1))
            max_int = 2**(bit_width - 1) - 1

        def quant_slice(input: torch.Tensor, s, z):
            assert (input.dim() == 1) and (s.dim() == 1)
            quantized = torch.clamp(
                torch.round(input / s) + z, min_int, max_int)
            dequantized = (quantized - z) * s
            return dequantized, quantized

        # Use transposed layout to speedup the memory access.
        quantized = torch.full((in_features, out_features),
                               float('nan'),
                               dtype=dtype,
                               device=device)
        # if reorder and (not jit_group_param):
        #     _, scales, zeros = cls.__quantize_weight_rtn(weight,
        #                                                  activation,
        #                                                  bit_width,
        #                                                  group_size,
        #                                                  asymmetric,
        #                                                  enable_clip_shrink,
        #                                                  fake=False)
        #     scales = scales.reshape(out_features,
        #                             num_groups_per_channel).t().contiguous()
        #     if asymmetric:
        #         zeros = zeros.reshape(out_features,
        #                               num_groups_per_channel).t().contiguous()
        # else:
        #     if jit_group_param:
        #         param_ready = set()
        scales = torch.full((num_groups_per_channel, out_features),
                            float('nan'),
                            dtype=dtype,
                            device=device)
        if asymmetric:
            zeros = torch.full((num_groups_per_channel, out_features),
                               float('nan'),
                               dtype=dtype,
                               device=device)
        input = activation.reshape(-1, in_features)
        for i1 in range(0, in_features, block_size):
            i2 = min(i1 + block_size, in_features)
            count = i2 - i1

            weight_block_t = weight_update_t[i1:i2, :].clone().contiguous()
            err_block_t = torch.zeros_like(weight_block_t)
            Hinv_block = Hinv[i1:i2, i1:i2]

            # We can do this here because block_size == group_size.

            # if not reorder:
            if enable_clip_shrink:
                input_block = input[:, i1:i2].clone().to(
                    weight_block_t.dtype).contiguous()
                s, z, _, _ = cls.__search_clip_param_groupwise(
                    weight_block_t.t().contiguous(), input_block, bit_width,
                    group_size, asymmetric)
            else:
                s, z, _, _ = cls.__get_quant_param(
                    weight_block_t.t().contiguous(), bit_width, group_size,
                    asymmetric)
            s = s.reshape(-1)
            z = z.reshape(-1) if not isinstance(z, int) else z

            for i in range(count):
                feature_idx = i1 + i
                # if reorder:
                #     feature_idx_real = perm[feature_idx]
                #     if jit_group_param:
                #         group_idx_real = feature_idx_real // group_size
                #         if group_idx_real not in param_ready:
                #             group_indices = list(
                #                 range(group_idx_real * group_size,
                #                       min((group_idx_real + 1) * group_size)))
                #             group_indices = perm_inv[torch.tensor(group_indices,
                #                                          device=weight.device)]
                #             weight_group_tensor = ...
                #             input_group_tensor = ...
                #     else:
                #         s = scales[feature_idx_real, :]
                #         if asymmetric:
                #             z = zeros[feature_idx_real, :]
                #         else:
                #             z = zeros

                w = weight_block_t[i, :]
                d = Hinv_block[i, i]
                w_q, q = quant_slice(w, s, z)
                # TODO: to wrap more functions into the compile scope.
                err_feature = (w - w_q) / d
                quantized[feature_idx] = q
                weight_block_t[i:, :] -= torch.einsum('i,j->ji', err_feature,
                                                      Hinv_block[i, i:])
                err_block_t[i, :] = err_feature
            weight_update_t[i2:, :] -= torch.einsum('bo,br->ro', err_block_t,
                                                    Hinv[i1:i2, i2:])

            group_idx = i1 // block_size
            scales[group_idx] = s.to(dtype).reshape(out_features)
            if asymmetric:
                zeros[group_idx] = z.to(dtype).reshape(out_features)

        if group_reorder:
            quantized = quantized[perm_inv, :]
            scales = scales[group_perm_inv, :]
            zeros = zeros[group_perm_inv, :] if asymmetric else 0

        quantized = quantized.t().contiguous()
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous() if asymmetric else 0
        del Hinv
        del weight_update_t

        if fake:
            dequantized = cls.__dequantize_weight_tensor(
                quantized, scales, zeros, group_size)
            return dequantized
        else:
            if bit_width == 8:
                dtype = torch.uint8 if asymmetric else torch.int8
            elif bit_width == 4:
                # To support the 4-bit, maybe by packing.
                dtype = torch.uint8 if asymmetric else torch.int8
            else:
                raise ValueError("Unsupported bit-width.")
            quantized = quantized.to(dtype).reshape(out_features, in_features)
            scales = scales.reshape(out_features, -1).t().contiguous()
            zeros = zeros.to(dtype).reshape(
                out_features, -1).t().contiguous() if asymmetric else zeros
            return quantized, scales, zeros

    @classmethod
    @torch.no_grad()
    def quantize_linear_weight(cls,
                               weight: torch.Tensor,
                               activation: torch.Tensor,
                               config: QuantConfig,
                               fake: bool = True,
                               info: str = None):
        config.verify(weight.shape)
        uniform_bit = config.get_uniform_bit(weight.shape)

        quantized = torch.empty_like(weight)

        if not fake:
            weight_int = {}
            weight_scale = {}
            weight_zero = {}
            weigth_indices = {}
        # Quantize different bit-widths separately.
        for bit_width, config_map in config.bit_config_map.items():
            group_size = config_map['group_size']
            asymmetric = config_map['asymmetric']
            # The weight's layout is [out_features, in_features]
            indices = config_map.get('indices', None)
            if (indices is None) and (uniform_bit == bit_width):
                # This means the weight is quantized uniformly.
                weight_chunk = weight
            elif indices:
                weight_chunk = weight[indices]
            else:
                # This means no quantization for this bit-width.
                continue

            is_gptq = config_map.get('gptq', False)
            enable_clip_shrink = config_map.get('clip_shrink', False)
            if is_gptq:
                group_reorder = config_map.get('gptq_group_reorder', True)
                try:
                    # reorder = False
                    # jit_group_param = False
                    if fake:
                        quantized_chunk = cls.__quantize_weight_gptq(
                            weight_chunk,
                            activation,
                            bit_width,
                            group_size,
                            asymmetric,
                            group_reorder,
                            # reorder,
                            # jit_group_param,
                            enable_clip_shrink,
                            fake=fake)
                    else:
                        weight_int[bit_width], weight_scale[bit_width], weight_zero[bit_width]\
                            = cls.__quantize_weight_gptq(
                                weight_chunk,
                                activation,
                                bit_width,
                                group_size,
                                asymmetric,
                                group_reorder,
                                # reorder,
                                # jit_group_param,
                                enable_clip_shrink,
                                fake=fake)
                        weigth_indices[bit_width] = indices
                except Exception as e:
                    logger.warning(f"Disable GPTQ for {info} due to {e}")
                    is_gptq = False
            if not is_gptq:
                assert fake == True, "Real quantization for RTN is not implemented yet."
                quantized_chunk = cls.__quantize_weight_rtn(weight_chunk,
                                                            activation,
                                                            bit_width,
                                                            group_size,
                                                            asymmetric,
                                                            enable_clip_shrink,
                                                            fake=fake)
        if fake:
            if cls.__DEBUG:
                assert quantized_chunk.isnan().sum() == 0
                assert quantized_chunk.isinf().sum() == 0
            if indices is None:
                quantized = quantized_chunk
            else:
                quantized[indices] = quantized_chunk
            return quantized
        else:
            return weight_int.get(8, None), weight_scale.get(8, None), weigth_indices.get(8, None), \
                   weight_int.get(4, None), weight_scale.get(4, None), \
                   weight_zero.get(4, None), weigth_indices.get(4, None)

    @classmethod
    def quantize_activation(cls,
                            activation: torch.Tensor,
                            bit_width: int,
                            group_size: int,
                            asymmetric: bool,
                            fake: bool = True,
                            info: str = None):
        assert fake == True
        # assert activation.dim() == 3
        hidden_size = activation.shape[-1]
        scales, zeros, min_int, max_int = cls.__get_quant_param(
            activation.reshape(-1, hidden_size), bit_width, group_size,
            asymmetric)

        return cls.__quantize_with_param(activation, bit_width, group_size,
                                         asymmetric, scales, zeros, min_int,
                                         max_int, fake)

    @classmethod
    def __dequantize_weight_tensor(cls, quantized, scales, zeros, group_size):
        out_features, in_features = quantized.shape
        if group_size > 0:
            assert in_features % group_size == 0
        else:
            group_size = in_features
        num_groups_per_channel = in_features // group_size
        group_number = out_features * num_groups_per_channel
        quantized = quantized.reshape(group_number, group_size)
        scales = scales.reshape(group_number, 1)
        zeros = zeros.reshape(group_number, 1) if isinstance(
            zeros, torch.Tensor) else zeros
        dequantized = (quantized - zeros) * scales
        return dequantized.reshape(out_features, in_features)

    @classmethod
    @torch.no_grad()
    def __qunatize_model_rtn(cls, model, mixllm_config: MixLLMConfig) -> None:
        weight_config = mixllm_config.weight_config
        assert len(weight_config) == 1

        bit_width = list(weight_config.keys())[0]
        config_map = weight_config[bit_width]
        group_size = config_map['group_size']
        asymmetric = config_map['asymmetric']

        named_linears = modeling.get_named_linears_in_transformer_layers(model)
        desc = 'Quant weight'
        tqdm_bar = tqdm(desc=desc, total=len(named_linears))
        for _, linear in named_linears:
            device = next(linear.parameters()).device
            weight_quant = cls.__quantize_weight_rtn(
                weight=linear.weight.cuda(),
                activation=None,
                bit_width=bit_width,
                group_size=group_size,
                asymmetric=asymmetric,
                fake=True)
            if cls.__DEBUG:
                assert weight_quant.isnan().sum() == 0
                assert weight_quant.isinf().sum() == 0
            linear.weight.data = weight_quant.data.to(device)
            tqdm_bar.update(1)
        tqdm_bar.close()

    @classmethod
    @torch.no_grad()
    def __quantize_model_advanced(cls, model, calib_input: torch.Tensor,
                                  mixllm_config: MixLLMConfig) -> None:
        named_linears = modeling.get_named_linears_in_transformer_layers(model)
        desc = 'Quant weight'
        tqdm_bar = tqdm(desc=desc, total=len(named_linears))

        if False:
            print(f"keys in map: {mixllm_config.linear_config_map.keys()}")

        @torch.no_grad()
        def swap_in_pre_hook(module, args):
            '''
            This will: 1) swap the module to the GPU; 2) quantize the weight and
            update the module inplace.
            TODO: update the config.
            '''
            module.cuda()

            # quantize the weight
            weight = module.weight
            input = args[0]
            name = module.name
            linear_config = mixllm_config.linear_config_map[name]
            if False:
                print(f"name of linear: {name}")

            fake = False
            if fake:
                weight_quant = cls.quantize_linear_weight(weight,
                                                          input,
                                                          linear_config,
                                                          fake=True,
                                                          info=name)
                module.weight.data = weight_quant.data

            else:
                matrix_B_int8, matrix_scale_int8, matrix_indices_int8, \
                matrix_B_int4, matrix_scale_int4, matrix_zero, matrix_indices_int4 \
                    = cls.quantize_linear_weight(weight,
                                                input,
                                                linear_config,
                                                fake=False,
                                                info=name)
                lmixllm = LinearMixLLM4vLLM(
                    matrix_B_int8=matrix_B_int8,
                    matrix_B_int4=matrix_B_int4,
                    matrix_scale_int8=matrix_scale_int8,
                    matrix_scale_int4=matrix_scale_int4,
                    matrix_zero_int4=matrix_zero,
                    matrix_indices_int8=matrix_indices_int8,
                    matrix_indices_int4=matrix_indices_int4,
                    bias=module.bias)
                modeling.recursive_setattr(model, name, lmixllm)

            gc.collect()
            torch.cuda.empty_cache()

            tqdm_bar.update(1)

        @torch.no_grad()
        def swap_out_post_hook(module, args, output):
            # Swap the module back to the CPU.
            module.cpu()

        handles = []
        for name, linear in named_linears:
            pre_handle = linear.register_forward_pre_hook(swap_in_pre_hook)
            handles.append(pre_handle)
            handle = linear.register_forward_hook(swap_out_post_hook)
            handles.append(handle)
            linear.name = name
        output_embeddings = modeling.get_output_embeddings(model)
        # The output embedding can be memory consuming for some models with
        # large vocabulary size. Skip it.
        stop_handle = output_embeddings.register_forward_pre_hook(
            _exit_fwd_pre_hook)

        # Place the layers except for the linears in decoders to GPU.
        modeling.move_non_decoder_linear_to_device(model, 'cuda')
        try:
            with torch.no_grad():
                model(input_ids=calib_input, use_cache=False)
        except _EarlyExitException:
            pass
        modeling.move_non_decoder_linear_to_device(model, 'cpu')

        stop_handle.remove()
        for handle in handles:
            handle.remove()
        gc.collect()
        tqdm_bar.close()

    @classmethod
    @torch.no_grad()
    def quantize_model_fake(cls, model, calib_input: torch.Tensor,
                            mixllm_config: MixLLMConfig) -> None:
        cls.__quantize_model_advanced(model, calib_input, mixllm_config)

        if mixllm_config.activation_config and (
                mixllm_config.activation_config['bit_width'] < 16):
            named_linears = modeling.get_named_linears_in_transformer_layers(
                model)
            for name, linear in tqdm(named_linears,
                                     desc='Quant activation',
                                     total=len(named_linears)):
                # quantize the activation
                need_bias = False
                if hasattr(linear, 'bias') and linear.bias is not None:
                    need_bias = True
                new_linear = QuantizedLinearLayer(
                    activation_config=mixllm_config.activation_config,
                    # the original Linear parameters:
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    bias=need_bias,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype)
                new_linear.weight.data = linear.weight.data
                if need_bias:
                    new_linear.bias.data = linear.bias.data
                modeling.recursive_setattr(model, name, new_linear)
                del linear
