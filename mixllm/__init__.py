import sys
import sysconfig
import torch
from pathlib import Path

current_dir = Path(__file__).resolve().parent
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")

so_file = str(current_dir / f"kernels{ext_suffix}")
so_files = [so_file]

print(so_files)

assert (
    len(so_files) == 1
), f"Expected one kernels_mixllm.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

from mixllm.nn.modules.linear import LinearMixLLM
from mixllm.nn.modules.linear_for_vllm import LinearMixLLM4vLLM
from mixllm.nn.modules.mixllm_config import MixLLMConfig
import mixllm.nn.modules.utils
import mixllm.nn.modules.ops
