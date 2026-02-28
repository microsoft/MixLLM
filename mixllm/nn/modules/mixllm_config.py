# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin


@dataclass
class MixLLMConfig(PushToHubMixin):
    quant_method: str = field(default="mixllm")
    ratio: float = field(default=1.0)
    config_file_name = "config.json"
    modules_to_not_convert: Optional[List] = None

    @classmethod
    def from_dict(cls, quant_config: Dict = {}):
        if not quant_config:
            quant_config = cls()
        else:
            quant_config = cls(**quant_config)

        return quant_config

    def to_dict(self):
        return {
            "ratio": self.ratio,
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "ratio": self.ratio,
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    def from_transformers_dict(self, transformers_dict: Dict):
        return {
            "quant_method":
                transformers_dict.get("quant_method"),
            "ratio":
                transformers_dict.get("ratio"),
            "modules_to_not_convert":
                transformers_dict.get("modules_to_not_convert"),
        }
