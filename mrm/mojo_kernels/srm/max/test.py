from max import functional as F
from max import random
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.tensor import Tensor
from max import driver
from max import driver

x = Tensor.constant([[1.0, -2.0], [-3.0, 4.0]], dtype=DType.float32, device=driver.CPU())
x = x.to(driver.Accelerator()) # Moves tensor to GPU
y = F.relu(x)
print (y)
