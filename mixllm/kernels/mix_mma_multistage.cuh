// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"
#include <iostream>
#include "torch/csrc/cuda/Stream.h"
#include "./mma_multistage_testbed.h"
#include <filesystem>
#include "./mix_mma_config.h"

#include "cutlass_extension/mq_mma_tensor_op_sm80.h"
namespace fs = std::filesystem;

namespace mixllm{
class LinearMixLLM {
public:
  LinearMixLLM(){
    checkCudaErrors(cudaStreamCreate(&local_stream_int4));
    checkCudaErrors(cudaStreamCreate(&local_stream_int8));
    checkCudaErrors(cudaEventCreate(&fork_stream_event));
    checkCudaErrors(cudaEventCreate(&local_gemm_event_int4));
    checkCudaErrors(cudaEventCreate(&local_gemm_event_int8));
    initializeHashTable();
    this->best_config_dir = std::filesystem::path(__FILE__).parent_path().string() + "/best_config"; //fs::temp_directory_path()/"best_configs";
    if(! fs::exists(this->best_config_dir))
        fs::create_directory(this->best_config_dir);
    // printHashTable();

  };

  ~LinearMixLLM(){
    cudaError_t err = cudaGetDeviceCount(nullptr);
    if (err == cudaSuccess) {
        checkCudaErrors(cudaStreamDestroy(local_stream_int4));
        checkCudaErrors(cudaStreamDestroy(local_stream_int8));
        checkCudaErrors(cudaEventDestroy(fork_stream_event));
        checkCudaErrors(cudaEventDestroy(local_gemm_event_int4));
        checkCudaErrors(cudaEventDestroy(local_gemm_event_int8));
    }
  }

  static void initialize() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
        instance = std::make_unique<LinearMixLLM>();
    }
  }

  static torch::Tensor run(
                torch::Tensor &matrix_C_computed,
                torch::Tensor &matrix_A,
                torch::Tensor &matrix_scale_act, 
                torch::Tensor &matrix_zero,
                torch::Tensor &matrix_scale_int8,
                torch::Tensor &matrix_scale_int4,
                torch::Tensor &matrix_indices_int8,
                torch::Tensor &matrix_indices_int4,
                torch::Tensor &matrix_B_int8,
                torch::Tensor &matrix_B_interleaved) {
    return instance->gemm_launcher(
        matrix_C_computed, matrix_A, matrix_scale_act, 
        matrix_zero, matrix_scale_int8, matrix_scale_int4, 
        matrix_indices_int8, matrix_indices_int4, 
        matrix_B_int8, matrix_B_interleaved);
  }
  
  torch::Tensor gemm_launcher(
                torch::Tensor &matrix_C_computed,
                torch::Tensor &matrix_A,
                torch::Tensor &matrix_scale_act, 
                torch::Tensor &matrix_zero,
                torch::Tensor &matrix_scale_int8,
                torch::Tensor &matrix_scale_int4,
                torch::Tensor &matrix_indices_int8,
                torch::Tensor &matrix_indices_int4,
                torch::Tensor &matrix_B_int8,
                torch::Tensor &matrix_B_interleaved);

private:
  static std::unique_ptr<LinearMixLLM> instance;
  static std::mutex instance_mutex;
  cudaStream_t local_stream_int4;
  cudaStream_t local_stream_int8;
  cudaEvent_t fork_stream_event;
  cudaEvent_t local_gemm_event_int4;
  cudaEvent_t local_gemm_event_int8;
  std::unordered_map<std::string, std::vector<int>> hash_table;
  std::unordered_map<std::string, std::vector<int>> hash_table_rm;
  std::unordered_map<std::string, int> config_map;
  std::unordered_map<std::string, int> config_map_rm;
  fs::path best_config_dir;

  std::string vectorToString(const std::vector<int>& vec) {
      std::ostringstream oss;
      for (size_t i = 0; i < vec.size(); ++i) {
          if (i != 0) {
              oss << ",";
          }
          oss << vec[i];
      }
      return oss.str();
  }

  void initializeHashTable() {
      int id = 1;
      for (const auto& vec : gemm_configs) {
          std::string key = vectorToString(vec);
          config_map[key] = id++;
          hash_table[key] = vec;
      }
      id = 1;
      for (const auto& vec : gemm_configs_rm) {
          std::string key = vectorToString(vec);
          config_map_rm[key] = id++;
          hash_table_rm[key] = vec;
      }
  }

  void printHashTable() {
      std::cout << "CM GEMM Configurations:" << std::endl;
      for (const auto& pair : hash_table) {
          std::cout << pair.first << ": ";
          for (const auto& val : pair.second) {
              std::cout << val << " ";
          }
          std::cout << config_map[pair.first] << std::endl;
          std::cout << std::endl;
      }

      std::cout << "RM GEMM Configurations:" << std::endl;
      for (const auto& pair : hash_table_rm) {
          std::cout << pair.first << ": ";
          for (const auto& val : pair.second) {
              std::cout << val << " ";
          }
          std::cout << config_map_rm[pair.first] << std::endl;
          std::cout << std::endl;
      }
  }

  // It requires weight in column major.
  template <int NumStages, 
            int BLOCKSIZE_M, int BLOCKSIZE_N,
            int TILESIZE_M,  int TILESIZE_N>
  torch::Tensor gemm(
                    torch::Tensor &matrix_C_computed,
                    torch::Tensor &matrix_A,
                    torch::Tensor &matrix_scale_act, 
                    torch::Tensor &matrix_zero,
                    torch::Tensor &matrix_scale_int8,
                    torch::Tensor &matrix_scale_int4,
                    torch::Tensor &matrix_indices_int8,
                    torch::Tensor &matrix_indices_int4,
                    torch::Tensor &matrix_B_int8,
                    torch::Tensor &matrix_B_interleaved);

  // It requires weight in column major.
  template <int NumStages, 
            int BLOCKSIZE_M, int BLOCKSIZE_N,
            int TILESIZE_M,  int TILESIZE_N>
  torch::Tensor gemm_rm(
                    torch::Tensor &matrix_C_computed,
                    torch::Tensor &matrix_A,
                    torch::Tensor &matrix_scale_act, 
                    torch::Tensor &matrix_zero,
                    torch::Tensor &matrix_scale_int8,
                    torch::Tensor &matrix_scale_int4,
                    torch::Tensor &matrix_indices_int8,
                    torch::Tensor &matrix_indices_int4,
                    torch::Tensor &matrix_B_int8,
                    torch::Tensor &matrix_B_interleaved);
};

template <int NumStages, 
          int BLOCKSIZE_M, int BLOCKSIZE_N,
          int TILESIZE_M,  int TILESIZE_N>
torch::Tensor LinearMixLLM::gemm(
                   torch::Tensor &matrix_C_computed,
                   torch::Tensor &matrix_A,
                   torch::Tensor &matrix_scale_act, 
                   torch::Tensor &matrix_zero,
                   torch::Tensor &matrix_scale_int8,
                   torch::Tensor &matrix_scale_int4,
                   torch::Tensor &matrix_indices_int8,
                   torch::Tensor &matrix_indices_int4,
                   torch::Tensor &matrix_B_int8,
                   torch::Tensor &matrix_B_interleaved){
  using ThreadblockShape = cutlass::gemm::GemmShape<BLOCKSIZE_M, BLOCKSIZE_N, 64>;
  using WarpShape = cutlass::gemm::GemmShape<TILESIZE_M, TILESIZE_N, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  //using MmaType_INT4 = cutlass::arch::OpMultiplyAddMixedInputUpcast;
  using MmaType_INT4 = cutlass::arch::OpMultiplyAddMixedAndShuffledInputUpcast;
  using MmaType_INT8 = cutlass::arch::OpMultiplyAddSaturate;

  // Define the MmaCore components
  using MmaCore_INT8 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB_INT8, LayoutB, ElementC, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp,
      NumStages, MmaType_INT8>;
  using MmaCore_INT4 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB_INT4, LayoutB, ElementC, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp,
      NumStages, MmaType_INT4>;
  
  extension_cpp::Testbed<MmaCore_INT8> mixllm_int8;
  extension_cpp::Testbed<MmaCore_INT4> mixllm_int4;

  // assert(mixllm_int8.sufficient());
  // assert(mixllm_int4.sufficient());

  int m = matrix_A.size(0);
  int k = matrix_A.size(1);
  int n = matrix_C_computed.size(0);
  int partial_n_int8 = matrix_B_int8.size(0);
  assert(k % 128 == 0);
  int num_group = k / 128;
  Options options(m, n, k, partial_n_int8);

  assert(matrix_C_computed.size(1) == m);
  assert(matrix_scale_act.size(0) == num_group && matrix_scale_act.size(1) == (m+1)/2*2);
  assert(matrix_zero.size(0) == num_group && matrix_zero.size(1) == (options.partial_n_int4+3)/4*4);
  assert(matrix_scale_int8.size(0) == num_group && matrix_scale_int8.size(1) == (options.partial_n_int8+1)/2*2);
  assert(matrix_scale_int4.size(0) == num_group && matrix_scale_int4.size(1) == (options.partial_n_int4+1)/2*2);
  assert(matrix_indices_int8.numel() == options.partial_n_int8);
  assert(matrix_indices_int4.numel() == options.partial_n_int4);
  assert(matrix_B_int8.size(0) == options.partial_n_int8 && matrix_B_int8.size(1) == k);
  assert(matrix_B_interleaved.size(0) == options.partial_n_int4 && matrix_B_interleaved.size(1) * 2 == k);
  
  dim3 block(32, (ThreadblockShape::kM / WarpShape::kM) * (ThreadblockShape::kN / WarpShape::kN), 1);

  int64_t logicalGridM = (int64_t(m) + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
  int64_t logicalGridN_INT8 = (int64_t(options.partial_n_int8) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
  int64_t logicalGridN_INT4 = (int64_t(options.partial_n_int4) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

  dim3 grid_int8(logicalGridM, logicalGridN_INT8);
  dim3 grid_int4(logicalGridM, logicalGridN_INT4);
  
  cudaStream_t stream_for_graph = at::cuda::getCurrentCUDAStream().stream();
  checkCudaErrors(cudaEventRecord(this->fork_stream_event, stream_for_graph));
  

  if(options.partial_n_int4 > 0){
    checkCudaErrors(cudaStreamWaitEvent(this->local_stream_int4, this->fork_stream_event, 0));
    mixllm_int4.run(grid_int4, block, options, matrix_A, matrix_B_interleaved, 
            matrix_scale_act, matrix_scale_int4, matrix_zero, matrix_indices_int4, matrix_C_computed, this->local_stream_int4);
    checkCudaErrors(cudaEventRecord(this->local_gemm_event_int4, this->local_stream_int4));
  }
  if(options.partial_n_int8 > 0){
    checkCudaErrors(cudaStreamWaitEvent(this->local_stream_int8, this->fork_stream_event, 0));
    mixllm_int8.run(grid_int8, block, options, matrix_A, matrix_B_int8, 
            matrix_scale_act, matrix_scale_int8, matrix_zero, matrix_indices_int8, matrix_C_computed, this->local_stream_int8);
    checkCudaErrors(cudaEventRecord(this->local_gemm_event_int8, this->local_stream_int8));
  }

  checkCudaErrors(cudaStreamWaitEvent(stream_for_graph, this->local_gemm_event_int4, 0));
  checkCudaErrors(cudaStreamWaitEvent(stream_for_graph, this->local_gemm_event_int8, 0));
  
  return matrix_C_computed;
}

template <int NumStages, 
          int BLOCKSIZE_M, int BLOCKSIZE_N,
          int TILESIZE_M,  int TILESIZE_N>
torch::Tensor LinearMixLLM::gemm_rm(
                   torch::Tensor &matrix_C_computed,
                   torch::Tensor &matrix_A,
                   torch::Tensor &matrix_scale_act, 
                   torch::Tensor &matrix_zero,
                   torch::Tensor &matrix_scale_int8,
                   torch::Tensor &matrix_scale_int4,
                   torch::Tensor &matrix_indices_int8,
                   torch::Tensor &matrix_indices_int4,
                   torch::Tensor &matrix_B_int8,
                   torch::Tensor &matrix_B_interleaved){
  using ThreadblockShape = cutlass::gemm::GemmShape<BLOCKSIZE_M, BLOCKSIZE_N, 64>;
  using WarpShape = cutlass::gemm::GemmShape<TILESIZE_M, TILESIZE_N, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  //using MmaType_INT4 = cutlass::arch::OpMultiplyAddMixedInputUpcast;
  using MmaType_INT4 = cutlass::arch::OpMultiplyAddMixedAndShuffledInputUpcast;
  using MmaType_INT8 = cutlass::arch::OpMultiplyAddSaturate;

  // Define the MmaCore components
  using MmaCore_INT8 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB_INT8, LayoutB, ElementC, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
      NumStages, MmaType_INT8>;
  using MmaCore_INT4 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB_INT4, LayoutB, ElementC, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
      NumStages, MmaType_INT4>;
  
  extension_cpp::Testbed<MmaCore_INT8, true> mixllm_int8;
  extension_cpp::Testbed<MmaCore_INT4, true> mixllm_int4;

  // assert(mixllm_int8.sufficient());
  // assert(mixllm_int4.sufficient());

  int m = matrix_A.size(0);
  int k = matrix_A.size(1);
  int n = matrix_C_computed.size(1);
  int partial_n_int8 = matrix_B_int8.size(0);
  assert(k % 128 == 0);
  int num_group = k / 128;
  Options options(m, n, k, partial_n_int8);

  assert(matrix_C_computed.size(0) == m);
  assert(matrix_scale_act.size(0) == num_group && matrix_scale_act.size(1) == (m+1)/2*2);
  assert(matrix_zero.size(0) == num_group && matrix_zero.size(1) == (options.partial_n_int4+3)/4*4);
  assert(matrix_scale_int8.size(0) == num_group && matrix_scale_int8.size(1) == (options.partial_n_int8+1)/2*2);
  assert(matrix_scale_int4.size(0) == num_group && matrix_scale_int4.size(1) == (options.partial_n_int4+1)/2*2);
  assert(matrix_indices_int8.numel() == options.partial_n_int8);
  assert(matrix_indices_int4.numel() == options.partial_n_int4);
  assert(matrix_B_int8.size(0) == options.partial_n_int8 && matrix_B_int8.size(1) == k);
  assert(matrix_B_interleaved.size(0) == options.partial_n_int4 && matrix_B_interleaved.size(1) * 2 == k);
  
  dim3 block(32, (ThreadblockShape::kM / WarpShape::kM) * (ThreadblockShape::kN / WarpShape::kN), 1);

  int64_t logicalGridM = (int64_t(m) + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
  int64_t logicalGridN_INT8 = (int64_t(options.partial_n_int8) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
  int64_t logicalGridN_INT4 = (int64_t(options.partial_n_int4) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

  dim3 grid_int8(logicalGridM, logicalGridN_INT8);
  dim3 grid_int4(logicalGridM, logicalGridN_INT4);
  
  cudaStream_t stream_for_graph = at::cuda::getCurrentCUDAStream().stream();
  checkCudaErrors(cudaEventRecord(this->fork_stream_event, stream_for_graph));
  

  if(options.partial_n_int4 > 0){
    checkCudaErrors(cudaStreamWaitEvent(this->local_stream_int4, this->fork_stream_event, 0));
    mixllm_int4.run(grid_int4, block, options, matrix_A, matrix_B_interleaved, 
            matrix_scale_act, matrix_scale_int4, matrix_zero, matrix_indices_int4, matrix_C_computed, this->local_stream_int4);
    checkCudaErrors(cudaEventRecord(this->local_gemm_event_int4, this->local_stream_int4));
  }
  if(options.partial_n_int8 > 0){
    checkCudaErrors(cudaStreamWaitEvent(this->local_stream_int8, this->fork_stream_event, 0));
    mixllm_int8.run(grid_int8, block, options, matrix_A, matrix_B_int8, 
            matrix_scale_act, matrix_scale_int8, matrix_zero, matrix_indices_int8, matrix_C_computed, this->local_stream_int8);
    checkCudaErrors(cudaEventRecord(this->local_gemm_event_int8, this->local_stream_int8));
  }

  checkCudaErrors(cudaStreamWaitEvent(stream_for_graph, this->local_gemm_event_int4, 0));
  checkCudaErrors(cudaStreamWaitEvent(stream_for_graph, this->local_gemm_event_int8, 0));
  
  return matrix_C_computed;
}



torch::Tensor LinearMixLLM::gemm_launcher(
                   torch::Tensor &matrix_C_computed,
                   torch::Tensor &matrix_A,
                   torch::Tensor &matrix_scale_act, 
                   torch::Tensor &matrix_zero,
                   torch::Tensor &matrix_scale_int8,
                   torch::Tensor &matrix_scale_int4,
                   torch::Tensor &matrix_indices_int8,
                   torch::Tensor &matrix_indices_int4,
                   torch::Tensor &matrix_B_int8,
                   torch::Tensor &matrix_B_interleaved){
    int M = matrix_A.size(0);
    int N = matrix_indices_int8.numel() + matrix_indices_int4.numel();
    int K = matrix_A.size(1);
    int N_partial = matrix_B_int8.size(0) / N_PARTIAL_BIN_SIZE;

    bool row_major = matrix_indices_int8.numel() == 0 || matrix_indices_int4.numel() == 0;

    int best_config_id = -1;
    int best_stage = -1;
    torch::Tensor output;

    // for row major output
    // i.e., W8A8 and W4A8 gemm
    if(row_major){
      std::ostringstream ss;
      ss << "RM_MBin-" << (M + M_BIN_SIZE - 1)/M_BIN_SIZE << "_N-" << N << "_K-" << K << "_8bit-" << N_partial;
      // std::cout << "Searching for best configuration for " << ss.str() << std::endl;
      if(fs::exists(this->best_config_dir/ss.str())){
        auto best_config_file = std::ifstream(this->best_config_dir/ss.str());
        best_config_file >> best_stage >> best_config_id;
      }
      
      if(M <= MAX_M_FOR_SEARCH && best_config_id < 0){
        torch::Tensor matrix_C_computed_ref;
        if(matrix_indices_int4.numel() == 0){ // int8 reference
          std::cout << "Will check reference." << std::endl;
          matrix_C_computed_ref = (matrix_A.to(torch::kFloat).view({M, K/128, 128}) 
                                    * (matrix_scale_act.view({K/128, (M+1)/2*2}).slice(1, 0, M).t().view({M, K/128, 1})))
                                  .view({M, K})
                                  .mm((matrix_B_int8.t().view({K/128, 128, N}).to(torch::kFloat)
                                    * (matrix_scale_int8.view({K/128, (N+1)/2*2}).slice(1, 0, N).view({K/128, 1, N})))
                                    .view({K, N})).contiguous();
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cudaEvent_t begin, end;
        checkCudaErrors(cudaEventCreate(&begin));
        checkCudaErrors(cudaEventCreate(&end));
        float fast_time = 1e8;
        auto configs = gemm_configs_rm;
        std::string best_key = "";
        for(int stage: {5, 11}){
          for (auto& config : configs)
          {
              int config_id = config_map_rm[vectorToString(config)];
              for (int i = 0; i < 5; ++i)
              {
                  CALL_GEMM_SWITCH_RM(stage, config_id)
              }

              // check reference
              if(matrix_indices_int4.numel() == 0){ // int8 reference
                // std::cout << "Checking reference" << std::endl;
                auto mismatch = (matrix_C_computed_ref - matrix_C_computed).abs() > 2e-3 + 3e-2 * matrix_C_computed_ref.abs();
                auto ratio = (mismatch.sum() / mismatch.numel()).item().toFloat();
                if( ratio > 1e-3){
                  std::cout << "Reference failed: " << ratio << std::endl;
                }
              }

              checkCudaErrors(cudaEventRecord(begin, stream));
              for (int i = 0; i < 10; ++i)
              {
                  CALL_GEMM_SWITCH_RM(stage, config_id)
              }
              checkCudaErrors(cudaEventRecord(end, stream));
              checkCudaErrors(cudaEventSynchronize(end));
              float time;
              checkCudaErrors(cudaEventElapsedTime(&time, begin, end));
              if (time < fast_time)
              {
                  fast_time = time;
                  best_config_id = config_id;
                  best_stage = stage;
                  best_key = vectorToString(config);
              }
          }
        }
        if(best_config_id >= 0){
          auto best_config_file = std::ofstream(this->best_config_dir/ss.str());
          best_config_file << best_stage << " " << best_config_id << std::endl << best_key << std::endl;
          std::cout << "best configuration saved to " << this->best_config_dir << std::endl;
        }
      }
      
      if (M <= MAX_M_FOR_SEARCH && best_config_id >= 0){
        CALL_GEMM_SWITCH_RM(best_stage, best_config_id)
      }
      else{
        return this->gemm_rm<5, 64, 64, 32, 32>(
          matrix_C_computed, matrix_A, matrix_scale_act, 
          matrix_zero, matrix_scale_int8, matrix_scale_int4, 
          matrix_indices_int8, matrix_indices_int4, 
          matrix_B_int8, matrix_B_interleaved);
      }
    }
    
    // for column major output
    // i.e., W4.4A8 gemm 
    else{
      std::ostringstream ss;
      ss << "MBin-" << (M + M_BIN_SIZE - 1)/M_BIN_SIZE << "_N-" << N << "_K-" << K << "_8bit-" << N_partial;
      // std::cout << "Searching for best configuration for " << ss.str() << std::endl;
      if(fs::exists(this->best_config_dir/ss.str())){
        auto best_config_file = std::ifstream(this->best_config_dir/ss.str());
        best_config_file >> best_stage >> best_config_id;
      }

      if(M <= MAX_M_FOR_SEARCH && best_config_id < 0){
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cudaEvent_t begin, end;
        checkCudaErrors(cudaEventCreate(&begin));
        checkCudaErrors(cudaEventCreate(&end));
        float fast_time = 1e8;
        auto configs = gemm_configs;
        std::string best_key = "";
        for(int stage: {5, 11}){
          for (auto& config : configs)
          {
              int config_id = config_map[vectorToString(config)];
              for (int i = 0; i < 5; ++i)
              {
                  CALL_GEMM_SWITCH(stage, config_id)
              }
              checkCudaErrors(cudaEventRecord(begin, stream));
              for (int i = 0; i < 10; ++i)
              {
                  CALL_GEMM_SWITCH(stage, config_id)
              }
              checkCudaErrors(cudaEventRecord(end, stream));
              checkCudaErrors(cudaEventSynchronize(end));
              float time;
              checkCudaErrors(cudaEventElapsedTime(&time, begin, end));
              if (time < fast_time)
              {
                  fast_time = time;
                  best_config_id = config_id;
                  best_stage = stage;
                  best_key = vectorToString(config);
              }
          }
        }
        if(best_config_id >= 0){
          auto best_config_file = std::ofstream(this->best_config_dir/ss.str());
          best_config_file << best_stage << " " << best_config_id << std::endl << best_key << std::endl;
          std::cout << "best configuration saved to " << this->best_config_dir << std::endl;
        }
      }
      
      if (M <= MAX_M_FOR_SEARCH && best_config_id >= 0)
        CALL_GEMM_SWITCH(best_stage, best_config_id)
      else
        return this->gemm<5, 64, 128, 64, 32>(
          matrix_C_computed, matrix_A, matrix_scale_act, 
          matrix_zero, matrix_scale_int8, matrix_scale_int4, 
          matrix_indices_int8, matrix_indices_int4, 
          matrix_B_int8, matrix_B_interleaved);
    }
    return output;
                   
  }

}
