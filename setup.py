# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# Custom build_ext command to compile CUDA code using Makefile
class CustomBuildExt(build_ext):
    def run(self):
        # Compile CUDA code using the Makefile in mixllm/kernels
        kernel_dir = os.path.join("mixllm", "kernels")
        make_cmd = ["make", "-C", kernel_dir, "kernels"]
        try:
            subprocess.check_call(make_cmd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile CUDA code: {e}")

        # Run the standard build_ext
        super().run()

# Custom install command to ensure build_ext runs
class CustomInstall(install):
    def run(self):
        self.run_command("build_ext")
        super().run()

# Define the extension module (placeholder for the .so file)
# The actual .so file is built by the Makefile
cuda_extension = Extension(
    "mixllm.kernels",
    sources=[],  # No direct sources, as Makefile handles compilation
)

# Find all Python source files
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
            print(filename)
    return paths

# Include the compiled .so file and Python source files
extra_files = package_files("mixllm")
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
print(os.path.join("mixllm", f"kernels{ext_suffix}"))
extra_files.append(os.path.join("mixllm", f"kernels{ext_suffix}"))

setup(
    name="mixllm",
    version="0.1",
    author="Zhen Zheng",
    author_email="zhengzhen.z@qq.com",
    description="Mixed-precision quantization for LLMs",
    python_requires=">=3.8",
    packages=[
        "mixllm",
        "mixllm.evaluation",
        "mixllm.kernels",
        "mixllm.nn",
        "mixllm.nn.modules",
        "mixllm.quantization",
        "mixllm.utils",
    ],
    package_data={
        "mixllm.kernels": [f"mixllm/kernels{ext_suffix}"],
        "mixllm": extra_files,
    },
    ext_modules=[cuda_extension],
    cmdclass={
        "build_ext": CustomBuildExt,
        "install": CustomInstall,
    },
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ])
