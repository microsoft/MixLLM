#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

export TORCH_HOME ?= `python3 -c 'import torch;print(torch.__path__[0])'`

install:
	cd mixllm/kernels && make -j kernels
	cp mixllm/kernels/kernels_v2.cpython*.so ./pymixllm/src/pymixllm/kernels_mixllm.so
	pip install -e pymixllm/

test:
	python pymixllm/test/test_kernel.py 
	python pymixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=0.3
