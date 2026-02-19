from layout import Layout, LayoutTensor
from algorithm import vectorize, parallelize
from math import sqrt, exp, min as math_min, max as math_max
from memory import memset_zero, memcpy, UnsafePointer
from random import rand, randn

fn allocate_shared_tensor(shape: TensorShape, dtype: DType) -> Tensor[dtype]:
    """
    Allocate a tensor in shared memory using aligned allocation.
    This allows the tensor to be safely accessed by multiple threads.
    
    Args:
        shape: Shape of the tensor to allocate
        dtype: Data type of the tensor
        
    Returns:
        Tensor allocated in shared memory with proper alignment
    """
    var shared_tensor = LayoutTensor[
        dtype,
        shape,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    
    return shared_tensor


fn allocate_tensor_with_flag(shape: TensorShape, dtype: DType, use_shared: Bool) -> Tensor[dtype]:
    """
    Allocate a tensor, optionally in shared memory.
    
    Args:
        shape: Shape of the tensor
        dtype: Data type
        use_shared: If True, allocate in shared memory; otherwise use standard allocation
        
    Returns:
        Allocated tensor
    """
    if use_shared:
        return allocate_shared_tensor(shape, dtype)
    else:
        var tensor = Tensor[dtype](shape)
        memset_zero(tensor.data(), tensor.num_elements())
        return tensor
