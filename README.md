# MixLLM: Mixed-Precision Quantization for Efficient LLM Inference

This repository contains the source code for **MixLLM**, an innovative framework for mixed-precision quantization designed to accelerate Large Language Model (LLM) inference while maintaining high accuracy.
This work is based on the principles and findings presented in the paper [MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design](https://arxiv.org/abs/2412.14590).

Large Language Models (LLMs) are powerful but demand significant computational resources and memory, making their deployment challenging.
Quantization is a crucial technique to mitigate these issues by reducing the precision of model weights and activations.
MixLLM offers flexible quantization solutions, supporting the mix of 4-bit and 8-bit in a single linear layer (e.g., W4.4A8 where the weight of the model are quantized with 10% 8-bit and 90% 4-bit and the activation is quantized with group-wise 8-bit).

## 2. Getting Started: Using the `mixllm` Package

This section guides you through installing the `mixllm` package and running basic tests.

### Source Code Installation

To get started with MixLLM, clone the repository and install the required dependencies.
The installation will take several minutes.

```bash
git clone --recursive https://github.com/microsoft/MixLLM.git
cd MixLLM
pip install -r requirements.txt

pip install -e .
````

You can optionally use the Dockerfile to build a Docker image before starting the installation.

```bash
docker build -t mixllm .
docker run -it --name <container_name> --runtime nvidia --gpus all \
    -v <host_path>:<container_path> \
    --ipc=host \
    mixllm \
    bash
```

### Running Kernel Tests

You can test the core mixed-precision kernels by running `test_kernel.py`. This script allows you to experiment with various parameters to understand the performance characteristics.

```bash
python mixllm/test/test_kernel.py
```

Example usage:

```bash
# Run a basic test with default parameters
python mixllm/test/test_kernel.py

# Run a test with specific input dimensions and mix quantization ratio
python mixllm/test/test_kernel.py -m=512 -n=4096 -k=4096 -r=0.1

```

Adjust the `-m`, `-n`, `-k` (matrix dimensions), and `-r` parameters to explore the mix quantization ratio of INT8 from INT4. The `-r` parameter represents the ratio of INT8 quantization: if `r=1`, it means 100% INT8 quantization; if `r=0`, it means 100% INT4 quantization; and if `r=0.1`, it means INT8 quantization accounts for 10% and INT4 quantization accounts for 90% of the mixed precision.

## 3\. Reproducing MixLLM Algorithm with Fake Quantization

For researchers interested in reproducing and experimenting with the MixLLM algorithm in MixLLM, this section outlines the steps for quantization and simulating quantized inference ("fake" inference).

The MixLLM algorithm typically involves:

1.  **Calibration:** Analyzing a small dataset to determine optimal quantization parameters (e.g., scales, zero points, and bit-width allocation for different layers/groups).

2.  **Quantization:** Applying the determined parameters to convert floating-point model weights and activations into their lower-precision integer representations.

3.  **Fake Quantization (Simulation):** Performing inference using the quantized weights and activations, but simulating the full-precision arithmetic in floating-point to evaluate accuracy without requiring actual low-precision hardware. This is crucial for algorithm development and debugging.

Fortunately, all these steps can be executed in one shot using the provided `mixllm/evaluation/run.sh` script.

To reproduce the MixLLM algorithm:


```bash
# Execute the full MixLLM quantization and fake inference pipeline
# The `run.sh` script encapsulates the calibration, quantization, and fake quantization evaluation.
./mixllm/evaluation/run.sh

```

## 4\. Integration with vllm Framework
MixLLM is designed to seamlessly integrate with vLLM, a high-throughput and memory-efficient inference engine for LLMs. This section describes how to use MixLLM quantized models with vLLM and perform various tests.

To integrate a MixLLM quantized model with vLLM, you typically need to apply the below patches into vllm v0.9.0.
```bash
# Apply vLLM Patches: Ensure necessary patches are applied to vLLM for compatibility with MixLLM's quantization.
./apply_vllm_patche.sh

# Install vLLM Framework: Navigate to the vllm directory and install the framework.
cd vllm
pip install -e . # This installation might take approximately 10 minutes.

# Run Example Inference Demo
python xn_quant_sample.py
```

You can evaluate the quantitative accuracy and performance of MixLLM quantized models within the vLLM framework using the provided scripts.
```bash
# To assess the accuracy of the quantized models, run the GSM8K evaluation script:
./vllm/run_gsm8k.sh

# To benchmark the inference performance, use the run_benchmark.sh script:
./vllm/run_benchmark.sh
```

## Citation

If you use MixLLM for your research, please cite our [paper](https://arxiv.org/abs/2412.14590):

```bibtex
@misc{zheng2024mixllm,
      title={MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design}, 
      author={Zhen Zheng and Xiaonan Song and Chuanjie Liu},
      year={2024},
      eprint={2412.14590},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.14590}, 
}
```