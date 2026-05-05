
from std.math import ceildiv

from std.gpu import block_dim, block_idx, thread_idx
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

from std.utils.index import IndexList

fn _row_repeat_cpu(
    output: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    rhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    ctx: DeviceContextPtr,
):
    var vector_length = output.dim_size(0)
    for i in range(vector_length):
        var idx = IndexList[output.rank](i)
        var result = lhs.load[1](idx) + rhs.load[1](idx)
        output.store[1](idx, result)

fn _row_repeat_gpu(
    output: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    rhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 16
    var gpu_ctx = ctx.get_device_context()
    var vector_length = output.dim_size(0)

    @parameter
    fn vector_addition_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(length):
            var idx = IndexList[output.rank](Int(tid))
            var result = lhs.load[1](idx) + rhs.load[1](idx)
            output.store[1](idx, result)

    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)
    gpu_ctx.enqueue_function_experimental[vector_addition_gpu_kernel](
        vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@compiler.register("row_repeat")
struct VectorAddition:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor[rank=1],
        lhs: InputTensor[dtype = output.dtype, rank = output.rank],
        rhs: InputTensor[dtype = output.dtype, rank = output.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "cpu":
            _row_repeat_cpu(output, lhs, rhs, ctx)
        elif target == "gpu":
            _row_repeat_gpu(output, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)