#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=80000 -n=28672 -k=4096 -r=0.1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=80000 -n=28672 -k=4096 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=80000 -n=28672 -k=4096 -r=1

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=0.1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=2048 -n=4096 -k=14336 -r=0.1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=2048 -n=4096 -k=4096 -r=0.1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=4096 -n=1024 -k=4096 -r=0.1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=128 -n=512 -k=512 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=128 -n=1024 -k=1024 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=0.875

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=0.75
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=0.5
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=1

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=2 -n=4 -k=128 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=3 -n=5 -k=128 -r=0.25

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=2 -n=8 -k=128 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=2 -n=8 -k=128 -r=1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=1 -n=1 -k=128 -r=0.2

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel.py -m=32 -n=128 -k=512 -r=0.25

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=512 -n=4096 -k=4096 -r=0.75
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=512 -n=4096 -k=4096 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=512 -n=4096 -k=4096 -r=1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=2048 -n=4096 -k=14336 -r=0.75
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=2048 -n=4096 -k=4096 -r=0.75
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=4096 -n=1024 -k=4096 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=128 -n=512 -k=512 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=128 -n=1024 -k=1024 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=512 -n=4096 -k=4096 -r=0.875

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=0.75
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=0.5
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=1

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=2 -n=4 -k=128 -r=0.25
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=3 -n=5 -k=128 -r=0.25

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=2 -n=8 -k=128 -r=0
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=2 -n=8 -k=128 -r=1
CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=1 -n=1 -k=128 -r=0.2

CUDA_VISIBLE_DEVICES=0 python pymixllm/test/test_kernel_no_cudagraph.py -m=32 -n=128 -k=512 -r=0.25


