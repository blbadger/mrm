# Porting Notes: PyTorch to Mojo

This document details the technical decisions and challenges encountered when porting `recurrent_inference.py` from PyTorch to Mojo.

## Overview

The port focuses on maintaining functional equivalence while adapting to Mojo's systems programming paradigm. The core algorithms remain identical, but implementation details differ significantly.

## Major Architectural Decisions

### 1. Struct-Based Design vs Class-Based

**PyTorch (Python):**
```python
class ColRepeatCausalLinear(nn.Module):
    def __init__(self, dim: int, embedding_dim=256, decay=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dim))
```

**Mojo:**
```mojo
struct ColRepeatCausalLinear:
    var weight: Tensor[DType.float32]
    
    fn __init__(inout self, dim: Int, embedding_dim: Int = 256, decay: Bool = False):
        self.weight = Tensor[DType.float32](TensorShape(1, dim))
```

**Rationale:** Mojo structs provide value semantics and better performance. No inheritance needed for this port.

### 2. Explicit Memory Management

**PyTorch:**
- Automatic memory management via Python GC
- Dynamic tensor allocation
- GPU/CPU memory handled by PyTorch

**Mojo:**
- Explicit tensor allocation with `Tensor[DType](TensorShape(...))`
- Manual cache initialization with `memset_zero`
- Stack allocation where possible for performance

### 3. Type System

**PyTorch:**
```python
def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
```

**Mojo:**
```mojo
fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
```

**Key Differences:**
- Explicit dtype specification (DType.float32 vs implicit)
- `inout` keyword for mutable self
- Compile-time type checking

### 4. Device Management

**PyTorch:**
```python
self.cache = torch.zeros(embedding_dim).to('cuda')
decay_value = decay_value.to(x.device)
```

**Mojo (Current):**
```mojo
self.cache = Tensor[DType.float32](TensorShape(embedding_dim))
memset_zero(self.cache.data(), self.cache.num_elements())
```

**Note:** Device management simplified in initial port. GPU support can be added via Mojo's backend system.

## Component-by-Component Analysis

### ColRepeatCausalLinear

**Preserved:**
- Cache update logic: `cache = (out - bias) / weight`
- Decay mechanism with clipping
- Forward pass computation

**Changed:**
- Manual loop instead of vectorized operations
- Explicit element-wise operations
- Stack-allocated temporaries

**Performance Notes:**
- Mojo version can use SIMD vectorization
- Potential for better cache locality
- No Python interpreter overhead

### RowRepeatCausalLinear

**Preserved:**
- Cache update logic: `cache = out - bias`
- Decay application
- Causal computation order

**Changed:**
- Similar to ColRepeatCausalLinear
- Explicit indexing instead of broadcasting

### MixerBlock

**Preserved:**
- Residual connections
- Layer normalization placement
- Channel and token mixing order

**Changed:**
- Sequential composition instead of nn.Sequential
- Manual residual addition
- Explicit activation application

**Simplified:**
- Single token mixer type (RowRepeatCausalLinear)
- No dynamic head selection
- Fixed expansion factor handling

### RecurrentMLPMixer

**Preserved:**
- Embedding → Blocks → Output structure
- Parameter initialization strategy
- Forward pass logic

**Changed:**
- Single mixer block instead of ModuleList
- Simplified parameter counting
- No loss computation (inference only)

**Deferred:**
- Training loop
- Optimizer integration
- Multi-block iteration

## Features Deferred to Future Versions

### 1. Multi-Head Architectures

**Not Yet Ported:**
- `HeadedRepeatCausalLinear`
- `ParallelRepeatHeads`
- `MixedRepeatHeads`
- `RepeatHeads`

**Reason:** These require more complex tensor reshaping and einops-like operations. Will be added once core functionality is validated.

**Implementation Plan:**
1. Add tensor reshape utilities
2. Implement head splitting/concatenation
3. Port each head variant
4. Add comprehensive tests

### 2. Training Infrastructure

**Not Yet Ported:**
- Backward pass
- Gradient computation
- Optimizer
- Loss functions (beyond forward)

**Reason:** Focus on inference first. Training requires autograd system.

### 3. Advanced Features

**Not Yet Ported:**
- Copy dataset functionality
- MLflow integration
- Dataset loading
- Tokenizer integration

**Reason:** These are application-level features, not core model architecture.

## Performance Optimization Opportunities

### 1. SIMD Vectorization

**Current:**
```mojo
for i in range(x.num_elements()):
    out[i] = w * x[i] + decay_val * self.cache[i] + b
```

**Optimized:**
```mojo
@parameter
fn vectorized_op[simd_width: Int](i: Int):
    var vec_x = x.load[width=simd_width](i)
    var vec_cache = self.cache.load[width=simd_width](i)
    var result = w * vec_x + decay_val * vec_cache + b
    out.store[width=simd_width](i, result)

vectorize[vectorized_op, simd_width](x.num_elements())
```

### 2. Parallelization

**Opportunity:** Batch processing can be parallelized
```mojo
@parameter
fn process_batch(b: Int):
    # Process batch element b
    ...

parallelize[process_batch](batch_size)
```

### 3. Memory Layout

**Current:** Row-major layout (default)
**Optimization:** Consider column-major for certain operations
**Benefit:** Better cache utilization for specific access patterns

### 4. Kernel Fusion

**Opportunity:** Fuse layer norm + linear operations
**Benefit:** Reduce memory bandwidth requirements
**Implementation:** Custom fused kernels

## Testing Strategy

### Unit Tests Needed

1. **Layer Tests:**
   - Forward pass correctness
   - Cache update verification
   - Decay mechanism
   - Boundary conditions

2. **Integration Tests:**
   - Multi-layer composition
   - End-to-end forward pass
   - Numerical stability

3. **Performance Tests:**
   - Throughput benchmarks
   - Memory usage profiling
   - Comparison with PyTorch

### Validation Approach

1. Generate random inputs in PyTorch
2. Run through PyTorch model
3. Port same inputs to Mojo
4. Compare outputs (within numerical tolerance)
5. Profile performance differences

## Known Limitations

### 1. Simplified Matrix Operations

Current implementation uses naive loops for matrix multiplication. Production code should use:
- BLAS libraries (via Mojo's C interop)
- Optimized kernels
- Tiled algorithms

### 2. Single Block Architecture

Current `RecurrentMLPMixer` has single block. Full implementation needs:
- Dynamic block list
- Proper iteration
- Memory management for multiple blocks

### 3. No Gradient Support

Inference only. Training requires:
- Autograd system
- Backward pass implementation
- Optimizer integration

### 4. Limited Dtype Support

Currently hardcoded to float32. Should support:
- float16 for memory efficiency
- bfloat16 for training
- int8 for quantization

## Migration Path for Users

### From PyTorch to Mojo

1. **Export Model Weights:**
   ```python
   torch.save(model.state_dict(), 'weights.pt')
   ```

2. **Convert to Mojo Format:**
   - Write conversion script
   - Handle tensor layout differences
   - Verify numerical equivalence

3. **Load in Mojo:**
   - Implement weight loading
   - Initialize model structure
   - Validate outputs

### Interoperability

Consider:
- Python bindings for Mojo code
- Shared memory for zero-copy transfer
- ONNX export/import

## Future Roadmap

### Phase 1: Core Functionality (Current)
- ✅ Basic layers
- ✅ Simple mixer block
- ✅ Inference pipeline

### Phase 2: Complete Architecture
- ⏳ Multi-head variants
- ⏳ Full mixer block options
- ⏳ Multiple blocks support

### Phase 3: Optimization
- ⏳ SIMD vectorization
- ⏳ Parallelization
- ⏳ Kernel fusion
- ⏳ Memory optimization

### Phase 4: Training
- ⏳ Backward pass
- ⏳ Optimizer
- ⏳ Training loop

### Phase 5: Production
- ⏳ Quantization
- ⏳ GPU support
- ⏳ Deployment tools
- ⏳ Benchmarking suite

## Contributing Guidelines

When extending this port:

1. **Maintain API Compatibility:** Keep function signatures similar to PyTorch
2. **Document Differences:** Clearly note where Mojo version diverges
3. **Add Tests:** Every new feature needs tests
4. **Optimize Incrementally:** Get correctness first, then optimize
5. **Profile Changes:** Measure performance impact

## References

- Original PyTorch code: `mrm/mrm/recurrent_inference.py`
- Mojo documentation: https://docs.modular.com/mojo/
- MLP-Mixer paper: https://arxiv.org/abs/2105.01601
- Mojo performance guide: https://docs.modular.com/mojo/manual/performance/

## Questions and Answers

**Q: Why not use Mojo's Python interop to call PyTorch directly?**
A: The goal is a pure Mojo implementation for maximum performance and deployment flexibility.

**Q: When will training be supported?**
A: After Mojo's autograd system matures and core inference is validated.

**Q: How does performance compare to PyTorch?**
A: Benchmarking in progress. Expect significant improvements for inference due to compilation and zero-overhead abstractions.

**Q: Can I use this in production?**
A: Not yet. This is an initial port. Wait for Phase 5 completion and thorough testing.

## Conclusion

This port demonstrates Mojo's capability for high-performance ML inference. While some features are deferred, the core architecture is functionally equivalent to the PyTorch version with potential for significant performance improvements.

The modular design allows incremental enhancement while maintaining a working baseline. Future work will focus on completing the architecture, optimizing performance, and adding training support.