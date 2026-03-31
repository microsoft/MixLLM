### Motivation 
TLDR; LLM tasks frequently require large batches or offline processing with a focus on throughput, featuring prefix sharing where different prompts can share common prefixes. Current LLM inference engines primarily optimize for streaming requests and struggle with large batched tasks, using LRU-based caching that may prematurely evict shared KV contexts. The streaming-oriented scheduling approach and token-based batching limitations result in suboptimal hardware utilization and inefficient handling of requests with longer decoding steps. 
 
Detailed insights can be found in our arXiv preprint [BatchLLM](https://arxiv.org/abs/2412.03594). Concurrent work (BlendServe, [2411.16102](https://arxiv.org/abs/2411.16102)) has also reported similar findings. 
 
### Proposed Changes 
 
1. Extended the current `LLMEngine` and `LLM` to enable scheduling based on prefix sharing groups: 
   - `LLMEngine`: 
     1. Defined `prefix_cluster` to organize requests into groups. Requests within each group share the same prefix and are scheduled by group. Simplified multi-level prefixes to single-level prefixes through dynamic programming algorithms to reduce system complexity and kernel overhead. 
     2. Added support for input formats in the form `[[prefix],[context1,context2,...,contextm]]`, enabling quick construction of two-level prefix indices while avoiding overhead from operations such as hashing. 
   - `LLM`: 
     1. Added support for treating the `prefix` as an independent request and constructing the computational dependency of `prefix-context`. 
 
2. Implemented and optimized the prefix-shared Attention kernel with horizontal fusion to reduce tail effects and kernel launch overhead: 
   1. `attention/ops/`: 
       1. Implemented BatchLLM kernels in CUDA and ROCm from OpenAI Triton. 
       2. Added `ds_attn_*.py` files representing multiple kernels for different computation platforms. 
   2. `attention/backends/`:   
      1. Registered `backend` for ROCm (ROCM_BATCH_LLM) and CUDA (BATCH_LLM).  
      2. Integrated [`aotriton`](https://github.com/ROCm/aotriton) to optimize ROCm performance. 
        
  3. `worker/`: 
       1. `worker.py`: Added warmup when selecting the Triton kernel for BatchLLM (executed before the current CUDA Graph warmup) 
       2. `model_runner.py`: Enhanced the `build()` operation for `AttentionMetaData` to collect additional pointer tensor information for triton kernel. 

4.  `core/`:
       1.  `scheduler.py`: Add the Resource-aware token-batching scheduler to optimize the performance of chunked prefill, which allows the scheduler to collect and construct a queue in the order of decoding request -> prefill request -> extra prefill request, and to a certain extent, temporarily ignore the constraint of `max_num_seqs` when there is an opportunity to fill the chunk.

 

 
5. Miscellaneous: 
   1. `env.py`: Added the `VLLM_USE_BATCHLLM` environment variable to control the use of Triton kernel for BatchLLM in global prefix sharing scenarios. 
   2. `arg_utils.py`: Added the `--enable-ahead-of-prefix-clustering` argument to enable the prefix clustering feature. 
   3. `test_context_share.py`: Added a test script to demonstrate BatchLLM's performance improvements. 
 
### Experimental Results 
 
We evaluated the proposed BatchLLM against `--enable-prefix-caching` using the paper's benchmark. The results demonstrate that BatchLLM achieves a 2x speedup. Detailed experimental results are available in the [paper](https://arxiv.org/abs/2412.03594).