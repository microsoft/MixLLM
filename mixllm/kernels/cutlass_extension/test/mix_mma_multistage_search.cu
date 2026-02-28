// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"
#include <iostream>
#include <vector>
#include "pure_kernels/test/unit/gemm/threadblock/mma_multistage_testbed.h"

void check_reference(Options &options,
      cutlass::HostTensor<ElementA, LayoutA> &matrix_A,
      cutlass::HostTensor<ElementB_INT8, LayoutB> &matrix_B_int8,
      cutlass::HostTensor<ElementB_INT4, LayoutB> &matrix_B_int4,
      cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> &matrix_scale_act,
      cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> &matrix_scale_int8,
      cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> &matrix_scale_int4,
      cutlass::HostTensor<ElementZero,  cutlass::layout::RowMajor> &matrix_zero,
      cutlass::HostTensor<int, cutlass::layout::ColumnMajor> &matrix_indices_int8,
      cutlass::HostTensor<int, cutlass::layout::ColumnMajor> &matrix_indices_int4,
      cutlass::HostTensor<ElementOutput, LayoutOutput> &matrix_C_computed){

    cutlass::HostTensor<ElementOutput, LayoutOutput> matrix_C_reference(
        cutlass::make_Coord(options.problem_size.m(), options.problem_size.n()), false);
    cutlass::reference::host::TensorFill(matrix_C_reference.host_view());
    //
    // Calculate the reference
    //
    cutlass::NumericConverter<ElementOutput, int> convert_to_output; 
    cutlass::NumericConverter<ElementA, ElementB_INT4> convert_int4_to_int8; 

    for (int i = 0; i < options.problem_size.m(); ++i) {
      for (int j = 0; j < options.partial_n_int8; ++j) {
        int b_c_d_col = matrix_indices_int8.at({0, j});
        for (int k = 0; k < options.problem_size.k(); ++k) {
            matrix_C_reference.at({i, b_c_d_col}) +=
              convert_to_output(matrix_A.at({i, k}) * 
                                matrix_B_int8.at({k, j}))
                * matrix_scale_int8.at({k/128, j}) * matrix_scale_act.at({k/128, i});
        }
      }
    }

    for (int i = 0; i < options.problem_size.m(); ++i) {
      for (int j = 0; j < options.partial_n_int4; ++j) {
        int b_c_d_col = matrix_indices_int4.at({0, j});
        for (int k = 0; k < options.problem_size.k(); ++k) {
            matrix_C_reference.at({i, b_c_d_col}) +=
              convert_to_output(matrix_A.at({i, k}) * 
                                (convert_int4_to_int8(matrix_B_int4.at({k, j})) - matrix_zero.at({k/128,j})))
                * matrix_scale_int4.at({k/128, j}) * matrix_scale_act.at({k/128, i});
        }
      }
    }

    bool passed = true;
    if constexpr (cutlass::platform::is_same<ElementOutput, cutlass::half_t>::value){
      passed = cutlass::reference::host::TensorRelativelyEquals(
        matrix_C_computed.host_view(), matrix_C_reference.host_view(), ElementOutput(1), ElementOutput(100));
    }
    else{
      passed = cutlass::reference::host::TensorEquals(
        matrix_C_computed.host_view(), matrix_C_reference.host_view());
    }

    if(passed) {
      std::cout << "Passed!\n";
      std::stringstream fname;
      fname << "passed_gather_GEMM_scatter_fusion.txt";
      std::ofstream file(fname.str());
      file
        << "A:\n" << matrix_A.host_view() << "\n"
        << "B_int8:\n" << matrix_B_int8.host_view() << "\n"
        << "B_int4:\n" << matrix_B_int4.host_view() << "\n"
        << "scale_int8:\n" << matrix_scale_int8.host_view() << "\n"
        << "scale_int4:\n" << matrix_scale_int4.host_view() << "\n"
        << "scale_act:\n" << matrix_scale_act.host_view() << "\n"
        << "zero:\n" << matrix_zero.host_view() << "\n"
        << "indices_int8:\n" << matrix_indices_int8.host_view() << "\n"
        << "indices_int4:\n" << matrix_indices_int4.host_view() << "\n"
        << "Reference:\n"
        << matrix_C_reference.host_view() << "\n"
        << "Computed:\n"
        << matrix_C_computed.host_view() << "\n";
    }

    else {
      std::cout << "Failed!\n";

      std::stringstream fname;
      fname << "error_gather_GEMM_scatter_fusion.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A:\n" << matrix_A.host_view() << "\n"
        << "B_int8:\n" << matrix_B_int8.host_view() << "\n"
        << "B_int4:\n" << matrix_B_int4.host_view() << "\n"
        << "scale_int8:\n" << matrix_scale_int8.host_view() << "\n"
        << "scale_int4:\n" << matrix_scale_int4.host_view() << "\n"
        << "scale_act:\n" << matrix_scale_act.host_view() << "\n"
        << "zero:\n" << matrix_zero.host_view() << "\n"
        << "indices_int8:\n" << matrix_indices_int8.host_view() << "\n"
        << "indices_int4:\n" << matrix_indices_int4.host_view() << "\n"
        << "Reference:\n"
        << matrix_C_reference.host_view() << "\n"
        << "Computed:\n"
        << matrix_C_computed.host_view() << "\n";
    }
    if(options.partial_n_int8 > 0)
      assert(cutlass::reference::host::TensorNorm(matrix_C_reference.host_view()) > 0);
}

int main(int argc, const char ** argv) {
  Options options(argc, argv);
  int m_init = options.problem_size.m();
#ifdef MIX
  std::vector<float> ratios = {0.1};
#else
  std::vector<float> ratios = {0, 1};
#endif
  std::vector<std::pair<int, int>> KNs = {{128, 512}};//{{4096, 1024}, {4096, 4096}, {4096, 14336}, {14336, 4096}, {8192, 8192}, {8192, 28672}};
  
  for(std::pair<int, int> KN : KNs){
    options.problem_size.k() = KN.first;
    options.problem_size.n() = KN.second;
    for(float ratio : ratios){
      options.ratio = ratio;
      options.parse(argc, argv);
      for(options.problem_size.m() = m_init; options.problem_size.m() <= 512; options.problem_size.m() *= 4){
        using ThreadblockShape = cutlass::gemm::GemmShape<BLOCKSIZE_M, BLOCKSIZE_N, 64>;
        using WarpShape = cutlass::gemm::GemmShape<TILESIZE_M, TILESIZE_N, 64>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
        int64_t logicalGridN_INT8 = (int64_t(options.partial_n_int8) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
        int64_t logicalGridN_INT4 = (int64_t(options.partial_n_int4) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

        using MmaType_INT4 = cutlass::arch::OpMultiplyAddMixedInputUpcast;
        using MmaType_INT8 = cutlass::arch::OpMultiplyAddSaturate;

        int const Stages = NumStages;

        // Define the MmaCore components
        using MmaCore_INT8 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB_INT8, LayoutB, ElementC, LayoutOutput, cutlass::arch::OpClassTensorOp,
            Stages, MmaType_INT8>;
        using MmaCore_INT4 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB_INT4, LayoutB, ElementC, LayoutOutput, cutlass::arch::OpClassTensorOp,
            Stages, MmaType_INT4>;
        
        dim3 block(32, (ThreadblockShape::kM / WarpShape::kM) * (ThreadblockShape::kN / WarpShape::kN), 1);



        int64_t logicalGridM = (int64_t(options.problem_size.m()) + ThreadblockShape::kM - 1) / ThreadblockShape::kM;

        dim3 grid_int8(logicalGridM, logicalGridN_INT8);
        dim3 grid_int4(logicalGridM, logicalGridN_INT4);
        

        // prepare input data for gemm kernel
        cutlass::HostTensor<ElementA, LayoutA> matrix_A(
                    cutlass::make_Coord(options.problem_size.m(), options.problem_size.k()));
        cutlass::HostTensor<ElementB_INT8, LayoutB> matrix_B_int8(
                    cutlass::make_Coord(options.problem_size.k(), options.partial_n_int8));
        cutlass::HostTensor<ElementB_INT4, LayoutB> matrix_B_int4(
                    cutlass::make_Coord(options.problem_size.k(), options.partial_n_int4));
        cutlass::HostTensor<ElementB_INT4, LayoutB> matrix_B_interleaved(matrix_B_int4.extent());
        cutlass::HostTensor<ElementB_INT4, LayoutB> matrix_B_interleaved_tmp(matrix_B_int4.extent());

        cutlass::HostTensor<ElementOutput, LayoutOutput> matrix_C_computed(
                    cutlass::make_Coord(options.problem_size.m(), options.problem_size.n()));
        cutlass::HostTensor<int, cutlass::layout::ColumnMajor> matrix_indices_int8(cutlass::make_Coord(1, options.partial_n_int8));
        cutlass::HostTensor<int, cutlass::layout::ColumnMajor> matrix_indices_int4(cutlass::make_Coord(1, options.partial_n_int4));
        cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> matrix_scale_int8(
                    cutlass::make_Coord(options.problem_size.k()/128, ((options.partial_n_int8+1)/2)*2));
        cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> matrix_scale_int4(
                    cutlass::make_Coord(options.problem_size.k()/128, ((options.partial_n_int4+1)/2)*2));
        cutlass::HostTensor<ElementScale, cutlass::layout::RowMajor> matrix_scale_act(
                    cutlass::make_Coord(options.problem_size.k()/128, ((options.problem_size.m()+1)/2)*2));
        cutlass::HostTensor<ElementZero,  cutlass::layout::RowMajor> matrix_zero(
                    cutlass::make_Coord(options.problem_size.k()/128, ((options.partial_n_int4+3)/4)*4));

        int scope_max = 8, scope_min = -8;
        uint64_t seed = 8;

        cutlass::reference::host::TensorFillRandomUniform(matrix_A.host_view(), seed, scope_max, scope_min, 0);

        cutlass::reference::host::TensorFillRandomUniform(
            matrix_B_int8.host_view(), seed + 16, scope_max, scope_min, 0);
        cutlass::reference::host::TensorFillRandomUniform(
            matrix_B_int4.host_view(), seed + 17, 0, 15, 0);
        
        for(int i = 0; i < options.problem_size.k(); i ++){
          for(int j = 0; j < options.partial_n_int4; j ++){
            int i_sub = i % 32;
            if(i_sub >= 4 && i_sub < 8){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i+12,j});
            }
            else if(i_sub >= 8 && i_sub < 12){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i-4,j});
            }
            else if(i_sub >= 12 && i_sub < 16){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i+8,j});
            }
            else if(i_sub >= 16 && i_sub < 20){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i-8,j});
            }
            else if(i_sub >= 20 && i_sub < 24){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i+4,j});
            }
            else if(i_sub >= 24 && i_sub < 28){
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i-12,j});
            }
            else{
              matrix_B_interleaved.at({i,j}) = matrix_B_int4.at({i,j});
            }
            matrix_B_interleaved_tmp.at({i,j}) = matrix_B_interleaved.at({i,j});
          }
        }

        for(int i = 0; i < options.problem_size.k(); i += 8){
          for(int j = 0; j < options.partial_n_int4; j ++){
            matrix_B_interleaved.at({i+0,j}) = matrix_B_interleaved_tmp.at({i+0,j});
            matrix_B_interleaved.at({i+1,j}) = matrix_B_interleaved_tmp.at({i+4,j});
            matrix_B_interleaved.at({i+2,j}) = matrix_B_interleaved_tmp.at({i+1,j});
            matrix_B_interleaved.at({i+3,j}) = matrix_B_interleaved_tmp.at({i+5,j});
            matrix_B_interleaved.at({i+4,j}) = matrix_B_interleaved_tmp.at({i+2,j});
            matrix_B_interleaved.at({i+5,j}) = matrix_B_interleaved_tmp.at({i+6,j});
            matrix_B_interleaved.at({i+6,j}) = matrix_B_interleaved_tmp.at({i+3,j});
            matrix_B_interleaved.at({i+7,j}) = matrix_B_interleaved_tmp.at({i+7,j});
          }
        }
        
        std::vector<int> to_fill(options.problem_size.n()) ; // vector with ints.
        std::iota (std::begin(to_fill), std::end(to_fill), 0); // Fill with 0, 1, ...., n
        { // std::random_shuffle was deprecated in C++14 and removed in C++17
#ifdef MIX
          std::random_device make_seed;
          std::mt19937 source_of_randomness(make_seed());
          std::shuffle(to_fill.begin(), to_fill.end(), source_of_randomness);
#endif
        }
        memcpy(matrix_indices_int8.host_data(), to_fill.data(), options.partial_n_int8 * sizeof(int));
        memcpy(matrix_indices_int4.host_data(), to_fill.data()+options.partial_n_int8, options.partial_n_int4 * sizeof(int));

        cutlass::reference::host::TensorFillRandomUniform(
          matrix_scale_int8.host_view(),
          seed+1,
          ElementScale(2),
          ElementScale(-2),
          0); 
        cutlass::reference::host::TensorFillRandomUniform(
          matrix_scale_int4.host_view(),
          seed+2,
          ElementScale(2),
          ElementScale(-2),
          0); 
        cutlass::reference::host::TensorFillRandomUniform(
          matrix_scale_act.host_view(),
          seed+3,
          ElementScale(2),
          ElementScale(-2),
          0); 
        cutlass::reference::host::TensorFillRandomUniform(
          matrix_zero.host_view(),
          seed+4,
          ElementZero(15),
          ElementZero(0),
          0); 
        cutlass::reference::host::TensorFill(matrix_C_computed.host_view());

        matrix_A.sync_device();
        matrix_B_int8.sync_device();
        matrix_B_interleaved.sync_device();
        matrix_indices_int8.sync_device();
        matrix_indices_int4.sync_device();
        matrix_scale_int8.sync_device();
        matrix_scale_int4.sync_device();
        matrix_scale_act.sync_device();
        matrix_zero.sync_device();
        matrix_C_computed.sync_device();

        cudaStream_t stream_for_graph;
        checkCudaErrors(cudaStreamCreate(&stream_for_graph));
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;

        test::gemm::threadblock::Testbed<MmaCore_INT8, useEpilogue> mixllm_int8(options);
        test::gemm::threadblock::Testbed<MmaCore_INT4, useEpilogue> mixllm_int4(options);
        assert(mixllm_int8.sufficient());
        assert(mixllm_int4.sufficient());

        if(options.use_cudagraph)
          checkCudaErrors(cudaStreamBeginCapture(stream_for_graph, cudaStreamCaptureModeGlobal));
        mixllm_int4.run(grid_int4, block, matrix_A, matrix_B_interleaved, 
                matrix_scale_act, matrix_scale_int4, matrix_zero, matrix_indices_int4, matrix_C_computed, stream_for_graph);
        mixllm_int8.run(grid_int8, block, matrix_A, matrix_B_int8, 
                matrix_scale_act, matrix_scale_int8, matrix_zero, matrix_indices_int8, matrix_C_computed, stream_for_graph);
        mixllm_int8.wait_local_gemm_event(stream_for_graph);
        mixllm_int4.wait_local_gemm_event(stream_for_graph);
        
        if(options.use_cudagraph){
          checkCudaErrors(cudaStreamEndCapture(stream_for_graph, &graph));
          checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
          checkCudaErrors(cudaGraphLaunch(graph_exec, stream_for_graph));
        }
        //
        // Check error code
        //
        checkCudaErrors(cudaStreamSynchronize(stream_for_graph));

        matrix_C_computed.sync_host();
        if(cutlass::reference::host::TensorNorm(matrix_C_computed.host_view()) < 1e-7)
          continue;

        if(options.reference_check)
          check_reference(options, matrix_A, matrix_B_int8, matrix_B_int4, matrix_scale_act, 
            matrix_scale_int8, matrix_scale_int4, matrix_zero, matrix_indices_int8, matrix_indices_int4, matrix_C_computed);

        cudaDeviceSynchronize();

        //
        // Profiling
        //

        cudaEvent_t events[2];

        for (auto & event : events) {
          checkCudaErrors(cudaEventCreate(&event));
        }

        checkCudaErrors(cudaEventRecord(events[0], stream_for_graph));
        for (int iter = 0; iter < options.iterations; ++iter) {
          if(options.use_cudagraph){
            checkCudaErrors(cudaGraphLaunch(graph_exec, stream_for_graph));
          }
          else{
            mixllm_int8.run(grid_int8, block, matrix_A, matrix_B_int8, 
                  matrix_scale_act, matrix_scale_int8, matrix_zero, matrix_indices_int8, matrix_C_computed, stream_for_graph);
            mixllm_int4.run(grid_int4, block, matrix_A, matrix_B_interleaved, 
                  matrix_scale_act, matrix_scale_int4, matrix_zero, matrix_indices_int4, matrix_C_computed, stream_for_graph);
            mixllm_int8.wait_local_gemm_event(stream_for_graph);
            mixllm_int4.wait_local_gemm_event(stream_for_graph);
          }
        }

        checkCudaErrors(cudaEventRecord(events[1], stream_for_graph));
        checkCudaErrors(cudaEventSynchronize(events[1]));

        // Measure elapsed runtime
        float runtime_ms = 0;
        checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

        // Compute average runtime and GFLOPs.
        runtime_ms = runtime_ms / options.iterations;
        int64_t fmas = int64_t(options.problem_size.m()) * int64_t(options.problem_size.n()) * int64_t(options.problem_size.k());
        float gflops = double(fmas) * 2.0 / double(1.0e9) / (runtime_ms / 1000.0);

        std::cout << "stage=" << NumStages \
                  << ", BLOCK_M=" << BLOCKSIZE_M \
                  << ", BLOCK_N=" << BLOCKSIZE_N \
                  << ", TILE_M="  << TILESIZE_M \
                  << ", TILE_N="  << TILESIZE_N;
        std::cout << ", ratio = " << options.ratio \
                  << ", M = " << options.problem_size.m() << ", N = " << options.problem_size.n() << ", K = " << options.problem_size.k();
        std::cout << ", Runtime: " << runtime_ms << " ms";
        std::cout << ", TOPS: " << gflops / 1000 << " TOPS\n\n";


        // Cleanup
        for (auto event : events) {
          (void)cudaEventDestroy(event);
        }
        checkCudaErrors(cudaStreamDestroy(stream_for_graph));
        if(options.use_cudagraph)
          checkCudaErrors(cudaGraphExecDestroy(graph_exec));

      }
    }
    
  }
  return 0;
}