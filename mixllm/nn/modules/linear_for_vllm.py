# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from mixllm.nn.modules.linear import LinearMixLLM
from typing import Union, List


class LinearMixLLM4vLLM(LinearMixLLM):

    def __init__(self,
                 matrix_B_int8: Union[Tensor, None],
                 matrix_scale_int8: Union[Tensor, None],
                 matrix_indices_int8: Union[Tensor, None, List[int]],
                 matrix_B_int4: Union[Tensor, None],
                 matrix_scale_int4: Union[Tensor, None, List[int]],
                 matrix_zero_int4: Union[Tensor, None],
                 matrix_indices_int4: Union[Tensor, None],
                 bias=None) -> None:
        super().__init__(weight_int8=matrix_B_int8,
                         weight_int4=matrix_B_int4,
                         weight_scale_int8=matrix_scale_int8,
                         weight_scale_int4=matrix_scale_int4,
                         weight_zero_int4=matrix_zero_int4,
                         indices_int8=matrix_indices_int8,
                         indices_int4=matrix_indices_int4,
                         bias=bias)

        # below is to for vllm only
        if isinstance(matrix_indices_int8, list):
            matrix_indices_int8 = torch.tensor(matrix_indices_int8,
                                               dtype=torch.int,
                                               device='cuda')
        if isinstance(matrix_indices_int4, list):
            matrix_indices_int4 = torch.tensor(matrix_indices_int4,
                                               dtype=torch.int,
                                               device='cuda')
        matrix_B_int8 = torch.tensor(
            [], device='cuda') if matrix_B_int8 is None else matrix_B_int8
        matrix_scale_int8 = torch.tensor(
            [],
            device='cuda') if matrix_scale_int8 is None else matrix_scale_int8
        matrix_B_int4 = torch.tensor(
            [], device='cuda') if matrix_B_int4 is None else matrix_B_int4
        matrix_scale_int4 = torch.tensor(
            [],
            device='cuda') if matrix_scale_int4 is None else matrix_scale_int4
        matrix_zero_int4 = torch.tensor(
            [], device='cuda') if matrix_zero_int4 is None else matrix_zero_int4
        matrix_indices_int8 = torch.tensor(
            [], device='cuda'
        ) if matrix_indices_int8 is None else matrix_indices_int8
        matrix_indices_int4 = torch.tensor(
            [], device='cuda'
        ) if matrix_indices_int4 is None else matrix_indices_int4

        matrix_zero_int8 = torch.zeros_like(matrix_scale_int8).to(torch.uint8)
        qweight = torch.cat([matrix_B_int8, matrix_B_int4], dim=0).contiguous()
        scale = torch.cat([matrix_scale_int8, matrix_scale_int4],
                          dim=-1).contiguous()
        zero = torch.cat([matrix_zero_int8, matrix_zero_int4],
                         dim=-1).contiguous()
        indices = torch.cat([matrix_indices_int8, matrix_indices_int4],
                            dim=-1).contiguous()
        is_int8_channel = torch.zeros_like(indices).bool()
        is_int8_channel[0:matrix_indices_int8.numel()] = 1
        is_int4_channel = is_int8_channel.logical_not()

        assert (is_int8_channel.sum().item() == matrix_indices_int8.numel())
        assert (is_int4_channel.sum().item() == matrix_indices_int4.numel())

        self.qweight = Parameter(qweight, requires_grad=False)

        self.scale = Parameter(scale, requires_grad=False)

        self.zero = Parameter(zero, requires_grad=False)

        self.indices = Parameter(indices, requires_grad=False)

        self.is_int8_channel = Parameter(is_int8_channel, requires_grad=False)

    def forward(self, x: Tensor):
        return super().forward(x)
