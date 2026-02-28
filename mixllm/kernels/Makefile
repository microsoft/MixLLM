# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

CONDA_HOME = /opt/conda
TORCH_HOME=$(shell python3 -c "import torch;print(torch.__path__[0])")

LDLIBS = -L${TORCH_HOME}/lib -L/usr/local/cuda/targets/x86_64-linux/lib/ -ltorch -ltorch_cpu -lc10 -ldl -lcudart -ltorch_python
LDLIBS += -L${CONDA_HOME}/lib -lpython3.11

LINUX_ARCH = $(shell uname -m)
PY_VERSION = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('py_version_nodot'))")
BUILD_DIR = ../../build/lib.linux-$(LINUX_ARCH)-cpython-$(PY_VERSION)/mixllm/

KERNELS = kernels
LIB_NAME = ${KERNELS}$(shell python3-config --extension-suffix)
ABI=$(shell python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

SOURCE = ./kernels.cu
DEPS = mix_mma_config.h mix_mma_multistage.cuh mma_multistage_testbed.h cutlass_extension/mq_default_epilogue_tensor_op.h cutlass_extension/mq_epilogue.h cutlass_extension/mq_fine_grained_scale_zero_iterator.h cutlass_extension/mq_mma_base.h cutlass_extension/mq_mma_mixed_input_tensor_op.h cutlass_extension/mq_mma_multistage.h cutlass_extension/mq_mma_tensor_op_dequantizer.h cutlass_extension/mq_mma_tensor_op_sm80.h cutlass_extension/mq_numeric_conversion.h

all: ${KERNELS}

${KERNELS}: ${SOURCE} ${DEPS}
	nvcc -shared -Xcompiler "-Wall -fPIC -fvisibility=hidden" \
		-I./cutlass/include \
		-I./cutlass/tools/util/include \
		-I${TORCH_HOME}/include/ \
		-I${TORCH_HOME}/include/torch/csrc/api/include/ \
		-I${CONDA_HOME}/include/python3.11/ \
		-I./cutlass_extension/ \
		${LDLIBS} \
		-D_GLIBCXX_USE_CXX11_ABI=${ABI} \
		-arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		-maxrregcount=255 \
		--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills \
		-lineinfo \
		-O3 --use_fast_math \
		-o ${LIB_NAME} ./kernels.cu

	mkdir -p ${BUILD_DIR}
	cp ${LIB_NAME} ${BUILD_DIR}
	cp ${LIB_NAME} ..

ARCH=80
CODEGEN_FLAGS=-gencode arch=compute_${ARCH},code=sm_${ARCH} -gencode arch=compute_${ARCH},code=compute_${ARCH}
FLAGS=${CODEGEN_FLAGS} -m64 --std=c++17 -lcublas -O3 -use_fast_math --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills -lineinfo  #--verbose -Xptxas -O3 -Xptxas -dlcm=cg -rdc=true

test_transpose: test_transpose.cu
	nvcc ${FLAGS} -o test_transpose test_transpose.cu

test_quantize: test_quantize.cu
	nvcc ${FLAGS} -o test_quantize test_quantize.cu


clean:
	rm ${LIB_NAME}

