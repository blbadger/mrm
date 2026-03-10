from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.tensor import Tensor, TensorType, defaults
from max.graph.type import Type
import max.functional as F
import max
import max.nn as nn
from max import driver
from max.driver import CPU, Device
from max.tensor import (
    Tensor,
    TensorType,
    default_device,
    default_dtype,
    defaults,
)

from max.nn import (
    Embedding,
    Linear,
    Module,
    Sequential,
)

import torch
from transformers import AutoTokenizer

import os 
from dotenv import load_dotenv
import pathlib
import time
from pathlib import Path

# custom_op_loader.py
import max.torch

# This handles compilation and registration
mojo_kernels = Path(__file__).parent / "kernels"
op_library = max.torch.CustomOpLibrary(mojo_kernels) 

def mojo_add_one(x):
    # Call the registered operation
    return op_library.ops.add_one(x)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        # Use your custom Mojo kernel as a layer
        x = mojo_add_one(x)
        return x

model = CustomModel()
input_type = TensorType((1, 10))
model.compile(input_type)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)

