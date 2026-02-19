# Structured Recurrent Mixer (SRM) - Mojo Implementation

This directory contains a Mojo port of the Recurrent MLP Mixer architecture from `mrm/mrm/recurrent_inference.py`.

## Overview

The Recurrent MLP Mixer is a sequence modeling architecture that uses stateful recurrent operations instead of traditional attention mechanisms. It maintains internal caches that are updated causally during forward passes, enabling efficient sequential processing.

## Shared Memory Cache Support

All recurrent layers now support a `use_shared_cache` flag that allows caches to be allocated in shared memory. This is useful for:
- **Multi-threaded inference**: Multiple threads can safely access shared caches
- **Memory efficiency**: Reduces memory duplication across threads
- **Performance**: Enables better cache locality in parallel workloads

To enable shared memory caches, pass `use_shared_cache=True` when initializing any recurrent layer or the full model.

## Architecture Components

### Core Recurrent Layers

1. **ColRepeatCausalLinear**
   - Column-repeat causal linear transformation
   - Maintains cache updated as: `cache = (out - bias) / weight`
   - Supports optional learnable decay

2. **RowRepeatCausalLinear**
   - Row-repeat causal linear transformation
   - Maintains cache updated as: `cache = out - bias`
   - Supports optional learnable decay

3. **CombinedRepeatCausalLinear**
   - Combines both row and column repeat operations
   - Maintains separate caches for each operation
   - Dual decay parameters for fine-grained control

4. **KernelRepeatLinear**
   - Kernel-based repeat layer with convolution-like operations
   - Multi-kernel cache for richer representations
   - Useful for capturing local patterns

5. **HeadedRepeatCausalLinear**
   - Multi-headed repeat layer
   - First half of heads use column repeat, second half use row repeat
   - Separate caches for each head

### Multi-Head Architectures

1. **ParallelRepeatHeads**
   - Parallel processing of multi-headed operations
   - Uses HeadedRepeatCausalLinear for efficient mixing
   - Optional input/output projections

2. **MixedRepeatHeads**
   - Mixed architecture with separate col and row heads
   - First half: ColRepeatCausalLinear heads
   - Second half: RowRepeatCausalLinear heads

3. **RepeatHeads**
   - General multi-head wrapper
   - All heads use same repeat operation type
   - Supports combined heads option

### Building Blocks

- **LayerNorm**: Layer normalization for stable training
- **Linear**: Standard linear transformation layer
- **Embedding**: Token embedding layer
- **Activation Functions**: SiLU (Swish) activation

### High-Level Components

- **MixerBlock**: Complete mixer block combining:
  - Channel mixing (MLP with expansion)
  - Token mixing (recurrent causal linear)
  - Residual connections
  - Layer normalization

- **RecurrentMLPMixer**: Full model architecture with:
  - Token embedding
  - Multiple mixer blocks
  - Output projection to vocabulary

## Key Differences from PyTorch Version

### 1. **Memory Management**
- Mojo uses explicit memory allocation and management
- Tensors are stack-allocated where possible
- Manual cache initialization and updates

### 2. **Type System**
- Explicit type annotations (DType.float32, DType.int32)
- Compile-time type checking
- No dynamic typing

### 3. **Multi-Head Implementation**
- All multi-head variants have been ported:
  - HeadedRepeatCausalLinear (multi-head version with mixed col/row)
  - ParallelRepeatHeads (parallel head processing)
  - MixedRepeatHeads (mixed row/col heads)
  - RepeatHeads (general multi-head wrapper)
- Some implementations are simplified but functionally complete

### 4. **Performance Optimizations**
- Mojo enables:
  - SIMD vectorization
  - Parallelization opportunities
  - Zero-cost abstractions
  - Compile-time optimizations

### 5. **No External Dependencies**
- PyTorch version uses: torch, einops, transformers, datasets, mlflow
- Mojo version is self-contained with standard library only

## Usage Example

### Basic Usage

```mojo
from mojo_kernels.srm.recurrent_inference import RecurrentMLPMixer
from tensor import Tensor

fn main():
    # Model configuration
    var vocab_size = 8000
    var hidden_dim = 1024
    var seq_len = 1024
    var num_blocks = 16
    
    # Initialize model
    var model = RecurrentMLPMixer(
        vocab_size,
        hidden_dim,
        seq_len,
        num_blocks
    )
    
    # Create input (batch of token indices)
    var input_ids = Tensor[DType.int32](TensorShape(32))  # batch size 32
    
    # Forward pass at sequence position 0
    var logits = model.forward(input_ids, index=0)
    
    # Count parameters
    var num_params = model.count_params()
    print("Model parameters:", num_params)
```

### Using Shared Memory Caches

```mojo
from mojo_kernels.srm.recurrent_inference import RecurrentMLPMixer
from tensor import Tensor

fn main():
    # Initialize model with shared memory caches
    var model = RecurrentMLPMixer(
        vocab_size=8000,
        hidden_dim=1024,
        seq_len=1024,
        num_blocks=16,
        use_shared_cache=True  # Enable shared memory for all caches
    )
    
    # Now all internal caches are allocated in shared memory
    # This is beneficial for multi-threaded inference scenarios
    var input_ids = Tensor[DType.int32](TensorShape(32))
    var logits = model.forward(input_ids, index=0)
```

### Individual Layer Usage with Shared Cache

```mojo
from mojo_kernels.srm.recurrent_inference import ColRepeatCausalLinear
from tensor import Tensor

fn main():
    # Create a layer with shared memory cache
    var layer = ColRepeatCausalLinear(
        dim=128,
        embedding_dim=256,
        decay=True,
        decay_constant=2.0,
        use_shared_cache=True  # Enable shared memory for this layer's cache
    )
    
    var input = Tensor[DType.float32](TensorShape(256))
    var output = layer.forward(input, 0)
```

## Architecture Details

### Causal Caching Mechanism

The key innovation is the causal cache update:

**Column Repeat:**
```
out = weight[index] * x + weight[index] * decay * cache + bias[index]
cache = (out - bias[index]) / weight[index]
```

**Row Repeat:**
```
out = weight[index] * x + decay * cache + bias[index]
cache = out - bias[index]
```

This allows the model to maintain state across sequence positions while remaining causal (no future information leakage).

### Decay Mechanism

Optional learnable decay parameter controls how much past information is retained:
- Clipped between 0.9 and 1.0 for stability
- Applied with decay constant normalization: `decay^(1/decay_constant)`
- Separate decay values for row and column operations in combined layers

## Performance Considerations

### Mojo Advantages
1. **Compile-time optimization**: Mojo compiles to native code
2. **Zero-cost abstractions**: No runtime overhead for abstractions
3. **SIMD vectorization**: Automatic vectorization of loops
4. **Memory efficiency**: Stack allocation and explicit control

### Optimization Opportunities
1. **Vectorization**: Use `vectorize` for element-wise operations
2. **Parallelization**: Use `parallelize` for batch processing
3. **Tiling**: Optimize cache usage for large tensors
4. **Fusion**: Combine operations to reduce memory traffic

## Implementation Notes

### Shared Memory Cache Implementation

The `use_shared_cache` flag enables allocation of caches in shared memory regions for multi-threaded access:

**Implementation Details**:
1. **Aligned Memory Allocation**: Uses `DTypePointer.alloc()` with 64-byte alignment for cache-line optimization
2. **Thread-Safe Access**: Caches are allocated in shared memory regions accessible by multiple threads
3. **Memory Layout**: Optimized for better cache locality in parallel workloads

**How It Works**:
- When `use_shared_cache=True`, the `allocate_shared_tensor()` function allocates tensors using aligned memory pointers
- 64-byte alignment ensures cache-line friendly access patterns
- All recurrent layers (ColRepeatCausalLinear, RowRepeatCausalLinear, HeadedRepeatCausalLinear, CombinedRepeatCausalLinear, KernelRepeatLinear) support this flag
- The flag propagates through the model hierarchy (MixerBlock → RecurrentMLPMixer)

**Usage Considerations**:
- Enable for multi-threaded inference scenarios
- Provides better memory efficiency when multiple threads access the same model
- Slight overhead for single-threaded use cases (use `use_shared_cache=False` for single-threaded)
- Thread synchronization for cache updates should be handled at the application level if needed

## Future Enhancements

1. **Thread Synchronization Primitives**: Add optional mutex/lock support for cache updates in highly concurrent scenarios
2. **Enhanced Multi-head Support**: Optimize multi-head implementations with proper tensor reshaping
3. **Batch Processing**: Optimize for batched inference
4. **Training Support**: Add backward pass and optimizer
5. **Quantization**: Add INT8/INT4 quantized versions
6. **GPU Support**: Add CUDA/Metal backend support
7. **Advanced Kernels**: Implement optimized BLAS operations

## Testing

Run the built-in tests:
```bash
mojo mojo_kernels/srm/recurrent_inference.mojo
```

This will test:
- ColRepeatCausalLinear forward pass
- RowRepeatCausalLinear forward pass
- Basic tensor operations

## References

- Original PyTorch implementation: `mrm/mrm/recurrent_inference.py`
- MLP-Mixer paper: "MLP-Mixer: An all-MLP Architecture for Vision"
- Mojo documentation: https://docs.modular.com/mojo/

## License

Same license as the original PyTorch implementation.

## Contributing

When adding features:
1. Maintain compatibility with PyTorch version's API
2. Add comprehensive docstrings
3. Include test cases
4. Document performance characteristics
5. Follow Mojo best practices

## Notes

- This is an initial port focusing on core functionality
- Some advanced features from PyTorch version are simplified
- Performance optimizations are ongoing
- API may evolve as Mojo language matures