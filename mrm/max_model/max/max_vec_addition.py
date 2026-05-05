# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# DOC: max/develop/get-started-with-max-graph-in-python.mdx

import numpy as np
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max import driver
from max.tensor import Tensor
import torch

def add_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 1. Build the graph
    input_type = TensorType(
        dtype=DType.float16, shape=(1,), device=driver.CPU()
    )
    with Graph(
        "simple_add_graph", input_types=(input_type, input_type)
    ) as graph:
        lhs, rhs = graph.inputs
        out = ops.add(lhs, rhs)
        graph.output(out)
        print("final graph:", graph)

    # 2. Create an inference session
    session = engine.InferenceSession([driver.CPU()])
    model = session.load(graph)

    for tensor in model.input_metadata:
        print(
            f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )

    # 3. Execute the graph
    output = model.execute(a, b)[0]
    print (f"Output: {output}")
    result = output.to_numpy()
    return result

def test_torch_tensor():
    input_tensor = torch.tensor([1.0], dtype=torch.float16)
    input0 = Tensor.constant(input_tensor, dtype=DType.float16, device=driver.CPU()).to(driver.Accelerator())
    input1 = Tensor.constant([1.0], dtype=DType.float16, device=driver.CPU()).to(driver.Accelerator())
    return input0, input1

if __name__ == "__main__":
    # external: take list, initialize as tensor on CPU, send to device
    
    #input0, input1 = test_torch_tensor()
    input0 = Tensor.constant([1.0], dtype=DType.float32, device=driver.CPU())
    input1 = Tensor.constant([1.0], dtype=DType.float32, device=driver.CPU())
    result = add_tensors(input0, input1)
    print("result:", result)
    assert result == [2.0]
