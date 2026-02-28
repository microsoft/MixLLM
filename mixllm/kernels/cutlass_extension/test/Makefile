# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

mix:
	nvcc -I./../cutlass/include \
		 -I./../cutlass/tools/util/include \
		 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		 -maxrregcount=255 \
		 --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills \
		 -lineinfo \
		 -O3 --use_fast_math \
		 -DMIX \
		 -DNumStages=6 \
		 -DBLOCKSIZE_M=16 -DBLOCKSIZE_N=32 \
		 -DTILESIZE_M=16 -DTILESIZE_N=16 \
		 -o mix_mma_multistage mix_mma_multistage.cu

int4:
	nvcc -I./pure_kernels/include \
		 -I./pure_kernels/tools/util/include \
		 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		 -maxrregcount=255 \
		 --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills \
		 -lineinfo \
		 -O3 --use_fast_math \
		 -DNumStages=6 \
		 -DBLOCKSIZE_M=16 -DBLOCKSIZE_N=32 \
		 -DTILESIZE_M=16 -DTILESIZE_N=16 \
		 -o mix_mma_multistage mix_mma_multistage.cu

int8:	
	nvcc -I./pure_kernels/include \
		 -I./pure_kernels/tools/util/include \
		 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		 -maxrregcount=255 \
		 --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills \
		 -lineinfo \
		 -O3 --use_fast_math \
		 -DNumStages=5 \
		 -DBLOCKSIZE_M=64 -DBLOCKSIZE_N=64 \
		 -DTILESIZE_M=32 -DTILESIZE_N=32 \
		 -o mix_mma_multistage mix_mma_multistage.cu

search:
	nvcc -I./pure_kernels/include \
		 -I./pure_kernels/tools/util/include \
		 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		 -maxrregcount=255 \
		 -O3 --use_fast_math \
		 -DNumStages=${NumStages} \
		 -DBLOCKSIZE_M=${BLOCKSIZE_M} -DBLOCKSIZE_N=${BLOCKSIZE_N} \
		 -DTILESIZE_M=${TILESIZE_M} -DTILESIZE_N=${TILESIZE_N} \
		 -o mix_mma_multistage_search mix_mma_multistage_search.cu

test:
	nvcc -I./pure_kernels/include \
		 -I./pure_kernels/tools/util/include \
		 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
		 -maxrregcount=255 \
		 --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills \
		 -lineinfo \
		 -O3 --use_fast_math \
		 -DNumStages=4 \
		 -DBLOCKSIZE_M=128 -DBLOCKSIZE_N=256 \
		 -DTILESIZE_M=32 -DTILESIZE_N=64 \
		 -o mix_mma_multistage mix_mma_multistage.cu

