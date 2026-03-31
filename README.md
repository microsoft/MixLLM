# BatchLLM

**Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching**

Large language models (LLMs) increasingly play an important role in a wide range of information processing and management tasks in industry. Many of these tasks are performed in large batches or even offline, where the key performance indicator is **throughput**. These workloads frequently exhibit **prefix sharing** — different prompt inputs partially share a common prefix.


BatchLLM achieves:

1. **Global prefix identification**: Explicitly identifies common prefixes across all requests globally. Requests sharing the same prefix are scheduled together to maximize KV context reuse.
2. **Elimination of redundant computation in inference**: By applying prefix-sharing groups, BatchLLM first infers and caches the KV states of the common prefix (as a single request without any decoding operation), then generates tokens for the remaining non-shared contexts/requests, saving redundant computation in both attention and non-attention parts for specific models.

> **Paper**: [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching]
>
> **Authors**: Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng

---

## Architecture

```
                         ┌────────────────────────────┐
                         │     Prompt Inputs           │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │    Prefix Clustering        │
                         │  (Global prefix detection   │
                         │   via dynamic programming)  │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │  Group-based Scheduling     │
                         │  (Decoding-first reorder +  │
                         │   Resource-aware batching)  │
                         └────────────┬───────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                   │
           ┌────────▼──────┐  ┌──────▼───────┐  ┌───────▼──────┐
           │ Common Prefix │  │   Distinct   │  │  Reduction   │
           │   Attention   │  │  Attention   │  │   Kernel     │
           └───────────────┘  └──────────────┘  └──────────────┘
```

### Key Components

| Component | Path | Description |
|-----------|------|-------------|
| **BatchLLM Backend** | `vllm/attention/backends/batch_llm.py` | Core attention backend with context sharing metadata |
| **Prefix-shared Kernel** | `vllm/attention/ops/ds_attn_*.py` | Triton JIT-compiled kernels with horizontal fusion |
| **Prefix Clustering** | `vllm/inputs/prefix_clustering.py` | Global prefix detection using prefix trees and DP |
| **CS Group Manager** | `vllm/core/csgroup_manager.py` | Context-sharing group lifecycle management |
| **Token Batching Scheduler** | `vllm/core/scheduler.py` | Resource-aware scheduler with decoding-first ordering |

---

## Getting Started

### Prerequisites

- NVIDIA GPU (CUDA)
- Docker

### Installation

**Step 1**: Pull the official vLLM 0.6.4 Docker image:

```bash
docker pull vllm/vllm-openai:v0.6.4
```

> Image available at: https://hub.docker.com/r/vllm/vllm-openai/tags?name=0.6.4

**Step 2**: Start a container from the image:
```
docker run -itd --shm-size 32g --gpus all -v [your mount path] --ipc=host --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --privileged --name vllm_064_retest --entrypoint="" vllm/vllm-openai:v0.6.4 sleep infinity
```

**Step 3**: Inside the container, clone this repository and replace the official vLLM with the BatchLLM branch:

```bash
git clone https://github.com/microsoft/MixLLM.git -b batchllm_vllm_064
cd batchllm_vllm_064
pip install -e .
```

**Step 4**: Run the test:

```bash
python3 test_context_share_064.py
```

---

## Usage

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_USE_BATCHLLM` | Set to `1` to enable Triton kernels for BatchLLM in global prefix sharing scenarios |
| `DISABLE_CS_KERNELS` | Set to `1` to disable context-sharing kernels (falls back to FlashAttention) |

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-ahead-of-prefix-clustering` | bool | `False` | Enable global prefix clustering as a preprocessing step |



## Disclaimer

**This is NOT an official Microsoft product.** This repository provides research and prototype code for academic and experimental purposes only. APIs, internal abstractions, and performance characteristics may change.

---

## License

This project is released under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for full details.

---

## Citation

waiting for the final version ready

---

## Contact

For questions related to the paper or this research prototype, please open an issue or contact the authors listed in the paper.
