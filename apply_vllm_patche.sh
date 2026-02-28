#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

set -e

echo "Applying vllm patches for version v0.9.0..."
cd vllm
git checkout releases/v0.9.0
for patch in ../vllm_v0.9.0_patch/*.patch; do
  git am "$patch"
done
echo "Patches applied successfully!"
