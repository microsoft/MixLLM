# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor, nn

import mixllm
from mixllm.nn.modules.utils import bcolors, time_logger


class LinearMixLLM(nn.Module):

    def __init__(self,
                 *kargs,
                 weight_int8,
                 weight_int4,
                 weight_scale_int8,
                 weight_scale_int4,
                 weight_zero_int4,
                 indices_int8,
                 indices_int4,
                 bias=None) -> None:
        super(LinearMixLLM, self).__init__()
        if isinstance(indices_int8, list):
            indices_int8 = torch.tensor(indices_int8,
                                        dtype=torch.int,
                                        device='cuda')
        if isinstance(indices_int4, list):
            indices_int4 = torch.tensor(indices_int4,
                                        dtype=torch.int,
                                        device='cuda')

        if weight_int4 is not None and weight_int8 is not None:
            assert (weight_int8.shape[1] == weight_int4.shape[1])
        if weight_int4 is not None:
            assert (weight_scale_int4.shape[0] == weight_zero_int4.shape[0])
            assert (weight_zero_int4.shape[1] == weight_int4.shape[0])
            assert (weight_scale_int4.shape[1] == weight_zero_int4.shape[1])
            assert (weight_scale_int4.shape[0] * 128 == weight_int4.shape[1])
            assert (indices_int4.numel() == weight_int4.shape[0])
            weight_int4 = weight_int4.cuda().contiguous()
            weight_scale_int4 = weight_scale_int4.cuda().contiguous()
            weight_zero_int4 = weight_zero_int4.cuda().contiguous()
            indices_int4 = indices_int4.cuda().contiguous()
        if weight_int8 is not None:
            assert (weight_scale_int8.shape[1] == weight_int8.shape[0])
            assert (weight_scale_int8.shape[0] * 128 == weight_int8.shape[1])
            assert (indices_int8.numel() == weight_int8.shape[0])
            weight_int8 = weight_int8.cuda().contiguous()
            weight_scale_int8 = weight_scale_int8.cuda().contiguous()
            indices_int8 = indices_int8.contiguous()

        partial_n_int8 = weight_int8.shape[0] if weight_int8 != None else 0
        partial_n_int4 = weight_int4.shape[0] if weight_int4 != None else 0
        self.in_features = weight_int8.shape[
            1] if weight_int8 != None else weight_int4.shape[1]
        self.out_features = partial_n_int8 + partial_n_int4
        num_groups = weight_scale_int8.shape[0] if weight_scale_int8 != None \
                    else weight_scale_int4.shape[0]

        if bias is not None:
            assert (self.out_features == bias.numel())

        # padding
        self.weight_scale_int8 = torch.empty(
            (num_groups, ((partial_n_int8 + 1) // 2) * 2),
            dtype=torch.float16,
            device="cuda:0")

        self.weight_scale_int4 = torch.empty(
            (num_groups, ((partial_n_int4 + 1) // 2) * 2),
            dtype=torch.float16,
            device="cuda:0")

        self.weight_zero_int4 = torch.empty(
            (num_groups, ((partial_n_int4 + 3) // 4) * 4),
            dtype=torch.uint8,
            device="cuda:0")

        if weight_int8 != None:
            self.weight_scale_int8[:, :weight_scale_int8.
                                   shape[1]] = weight_scale_int8.half().cuda(
                                   ).contiguous()
        if weight_int4 != None:
            self.weight_scale_int4[:, :weight_scale_int4.
                                   shape[1]] = weight_scale_int4.half().cuda(
                                   ).contiguous()
            self.weight_zero_int4[:, :weight_zero_int4.
                                  shape[1]] = weight_zero_int4.half().cuda(
                                  ).contiguous()

        self.indices_int8 = indices_int8.int().cuda().contiguous() \
                            if indices_int8 != None \
                            else torch.empty((partial_n_int8,), dtype=torch.int32, device="cuda:0")
        self.indices_int4 = indices_int4.int().cuda().contiguous() \
                            if indices_int4 != None \
                            else torch.empty((partial_n_int4,), dtype=torch.int32, device="cuda:0")

        self.weight_int8 = weight_int8.to(torch.int8).cuda().contiguous() \
                           if weight_int8 != None \
                           else torch.empty((partial_n_int8, self.in_features), dtype=torch.int8, device="cuda:0")
        self.weight_int4 = self.interleave_uint4_for_cutlass(weight_int4.to(torch.uint8).cuda()) \
                           if weight_int4 != None \
                           else torch.empty((partial_n_int4, self.in_features//2), dtype=torch.uint8, device="cuda:0")
        self.is_row_major = (partial_n_int8 == 0 or partial_n_int4 == 0)

        self.bias = bias

    def down_size_(self, size, scale):
        assert size[-1] % scale == 0, f"{size} last dim not divisible by {scale}"
        return (*size[:-1], size[-1] // scale)

    def pack_uint4_for_cutlass(self, uint8_data) -> torch.Tensor:
        # converting to uint8 for operations
        shape = uint8_data.shape
        assert shape[-1] % 2 == 0
        uint8_data = uint8_data.contiguous().view(-1)
        return (uint8_data[1::2] << 4 | uint8_data[::2]).view(
            self.down_size_(shape, 2)).contiguous()

    def interleave_uint4_for_cutlass(self, matrix_B_int4) -> torch.Tensor:
        matrix_B_interleaved = matrix_B_int4.clone()

        interleaved_index = []
        for i in range(matrix_B_int4.shape[1]):
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
        for i in range(0, matrix_B_int4.shape[1], 8):
            interleaved_index.append(i)
            interleaved_index.append(i + 4)
            interleaved_index.append(i + 1)
            interleaved_index.append(i + 5)
            interleaved_index.append(i + 2)
            interleaved_index.append(i + 6)
            interleaved_index.append(i + 3)
            interleaved_index.append(i + 7)
        matrix_B_interleaved = matrix_B_interleaved_tmp[:, interleaved_index]

        return self.pack_uint4_for_cutlass(matrix_B_interleaved)

    @torch.compile
    def activation_quantization(self, input: Tensor, act_scale_padded: Tensor):
        M = input.shape[0]
        act_scale = input.view(M, input.shape[1] // 128,
                               128).abs().amax(dim=-1) / 127
        input = input.view(M, input.shape[1] // 128, 128) / act_scale.view(
            act_scale.shape[0], act_scale.shape[1], 1)
        input_quantized = input.round().to(torch.int8).view(M, self.in_features)
        act_scale = act_scale.t().contiguous()
        act_scale_padded[:, :act_scale.shape[1]] = act_scale.half().cuda(
        ).contiguous()
        return input_quantized

    def forward(self, input: Tensor, return_quantization=False):
        assert (input.shape[-1] == self.in_features)
        input = input.view(-1, self.in_features)

        # """
        M = input.shape[0]

        if False:
            act_scale_padded = torch.ones((self.in_features // 128, M + M % 2),
                                          dtype=torch.float16,
                                          device="cuda:0")
            input_quantized = self.activation_quantization(
                input, act_scale_padded)
        else:
            input_quantized, act_scale_padded = mixllm.nn.modules.ops.quantize(
                input)
        """"
        # This is for debugging, it is the fake gemm implemented with torch
        input_dequant = input_quantized.to(torch.float16).view(M, self.in_features//128, 128) \
                * act_scale_padded.t()[0:M, :].view(M, self.in_features//128, 1)
        input_dequant = input_dequant.view(M, self.in_features)

        weight = self.weight_int8.t().to(torch.float16).contiguous().view(self.in_features//128, 128, self.out_features) \
                 * self.weight_scale_int8.view(self.in_features//128, 1, self.out_features)
        weight = weight.view(self.in_features, self.out_features)

        output = (input_dequant @ weight) + self.bias if self.bias is not None else input_dequant @ weight
        return output

        """
        output = mixllm.nn.modules.ops.mixllm_gemm(
            input_quantized, act_scale_padded, self.weight_zero_int4,
            self.weight_scale_int8, self.weight_scale_int4, self.indices_int8,
            self.indices_int4, self.weight_int8, self.weight_int4)
        if not self.is_row_major:
            # TODO: use our cuda transpose for W4.4A8 gemm implementation
            if (self.out_features == 4096 or self.out_features == 14336 or
                    self.out_features == 1024 or self.out_features == 6144 or
                    self.out_features == 28672):
                print(
                    bcolors.WARNING +
                    f"Use mixllm transpose. M = {M}, N = {self.out_features}" +
                    bcolors.ENDC)
                output = mixllm.nn.modules.ops.transpose(output)
            else:
                print(
                    bcolors.WARNING +
                    f"Use torch's transpose. Might be slow. M = {M}, N = {self.out_features}"
                    + bcolors.ENDC)
                output = output.t().contiguous()

        if self.bias is not None:
            output = output + self.bias

        if return_quantization:
            return output, input_quantized, act_scale_padded
        else:
            return output
