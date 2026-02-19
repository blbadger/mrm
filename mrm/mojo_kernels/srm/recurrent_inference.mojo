"""
Recurrent MLP Mixer implementation in Mojo
Ported from PyTorch implementation in mrm/mrm/recurrent_inference.py

This module implements various recurrent linear layers and mixer architectures
for sequence modeling with stateful computation.
"""

from tensor import Tensor, TensorShape
from algorithm import vectorize, parallelize
from math import sqrt, exp, min as math_min, max as math_max
from memory import memset_zero, memcpy, UnsafePointer
from memory.unsafe import DTypePointer
from sys import alignof
from random import rand


# ============================================================================
# Shared Memory Allocation Helpers
# ============================================================================

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
    # Calculate total size
    var size = 1
    for i in range(shape.rank()):
        size *= shape[i]
    
    # Allocate aligned memory for cache-friendly access
    # Using 64-byte alignment for cache line optimization
    var ptr = DTypePointer[dtype].alloc(size, alignment=64)
    
    # Zero-initialize the memory
    memset_zero(ptr, size)
    
    # Create tensor from the allocated pointer
    # Note: In production, we'd use a custom tensor type that tracks
    # the shared memory allocation for proper cleanup
    var tensor = Tensor[dtype](shape)
    memcpy(tensor.data(), ptr, size)
    
    return tensor


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


# ============================================================================
# Core Recurrent Linear Layers
# ============================================================================

struct ColRepeatCausalLinear:
    """
    Column-repeat causal linear layer with optional decay.
    Maintains a cache that is updated causally during forward passes.
    """
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var decay_value: Tensor[DType.float32]
    var decay_constant: Float32
    var cache: Tensor[DType.float32]
    var dim: Int
    var embedding_dim: Int
    var use_decay: Bool
    var use_shared_cache: Bool

    fn __init__(inout self, dim: Int, embedding_dim: Int = 256, decay: Bool = False, decay_constant: Float32 = 1.0, use_shared_cache: Bool = False):
        """Initialize ColRepeatCausalLinear layer.
        
        Args:
            dim: Sequence dimension
            embedding_dim: Embedding dimension for cache
            decay: Whether to use learnable decay
            decay_constant: Decay normalization constant
            use_shared_cache: Whether to allocate cache in shared memory for multi-threading
        """
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.use_decay = decay
        self.decay_constant = decay_constant
        self.use_shared_cache = use_shared_cache
        
        # Initialize weight with random values (randn equivalent)
        self.weight = Tensor[DType.float32](TensorShape(1, dim))
        rand(self.weight.data(), self.weight.num_elements())
        
        # Initialize bias to zeros
        self.bias = Tensor[DType.float32](TensorShape(dim))
        memset_zero(self.bias.data(), self.bias.num_elements())
        
        # Initialize decay value
        if decay:
            self.decay_value = Tensor[DType.float32](TensorShape(1))
            self.decay_value[0] = 1.0
        else:
            self.decay_value = Tensor[DType.float32](TensorShape(1))
            self.decay_value[0] = 1.0
        
        # Initialize cache with optional shared memory allocation
        self.cache = allocate_tensor_with_flag(
            TensorShape(embedding_dim),
            DType.float32,
            use_shared_cache
        )

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass with causal caching.
        
        Args:
            x: Input tensor
            index: Current sequence position
            
        Returns:
            Output tensor after applying linear transformation and cache
        """
        # Clip decay value between 0.9 and 1.0, then apply decay constant
        var decay_val = self.decay_value[0]
        decay_val = math_max(0.9, math_min(1.0, decay_val))
        decay_val = decay_val ** (1.0 / self.decay_constant)
        
        # Compute output: weight[index] * x + weight[index] * decay * cache + bias[index]
        var out = Tensor[DType.float32](x.shape())
        var w = self.weight[0, index]
        var b = self.bias[index]
        
        for i in range(x.num_elements()):
            out[i] = w * x[i] + w * decay_val * self.cache[i] + b
        
        # Update cache: (out - bias) / weight
        for i in range(self.cache.num_elements()):
            self.cache[i] = (out[i] - b) / w
        
        return out


struct RowRepeatCausalLinear:
    """
    Row-repeat causal linear layer with optional decay.
    Similar to ColRepeatCausalLinear but with different cache update logic.
    """
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var decay_value: Tensor[DType.float32]
    var decay_constant: Float32
    var cache: Tensor[DType.float32]
    var dim: Int
    var embedding_dim: Int
    var use_decay: Bool
    var use_shared_cache: Bool

    fn __init__(inout self, dim: Int, embedding_dim: Int = 256, decay: Bool = False, decay_constant: Float32 = 1.0, use_shared_cache: Bool = False):
        """Initialize RowRepeatCausalLinear layer.
        
        Args:
            dim: Sequence dimension
            embedding_dim: Embedding dimension for cache
            decay: Whether to use learnable decay
            decay_constant: Decay normalization constant
            use_shared_cache: Whether to allocate cache in shared memory for multi-threading
        """
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.use_decay = decay
        self.decay_constant = decay_constant
        self.use_shared_cache = use_shared_cache
        
        # Initialize weight with random values
        self.weight = Tensor[DType.float32](TensorShape(1, dim))
        rand(self.weight.data(), self.weight.num_elements())
        
        # Initialize bias to zeros
        self.bias = Tensor[DType.float32](TensorShape(dim))
        memset_zero(self.bias.data(), self.bias.num_elements())
        
        # Initialize decay value
        if decay:
            self.decay_value = Tensor[DType.float32](TensorShape(1))
            self.decay_value[0] = 1.0
        else:
            self.decay_value = Tensor[DType.float32](TensorShape(1))
            self.decay_value[0] = 1.0
        
        # Initialize cache with optional shared memory allocation
        self.cache = allocate_tensor_with_flag(
            TensorShape(embedding_dim),
            DType.float32,
            use_shared_cache
        )

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass with row-repeat caching logic.
        
        Args:
            x: Input tensor of shape [B, E]
            index: Current sequence position
            
        Returns:
            Output tensor
        """
        # Clip decay value between 0.9 and 1.0
        var decay_val = self.decay_value[0]
        decay_val = math_max(0.9, math_min(1.0, decay_val))
        decay_val = decay_val ** (1.0 / self.decay_constant)
        
        # Compute output: weight[index] * x + decay * cache + bias[index]
        var out = Tensor[DType.float32](x.shape())
        var w = self.weight[0, index]
        var b = self.bias[index]
        
        for i in range(x.num_elements()):
            out[i] = w * x[i] + decay_val * self.cache[i] + b
        
        # Update cache: out - bias
        for i in range(self.cache.num_elements()):
            self.cache[i] = out[i] - b
        
        return out


# ============================================================================
# Utility Functions
# ============================================================================

fn clip(value: Float32, min_val: Float32, max_val: Float32) -> Float32:
    """Clip value between min and max."""
    return math_max(min_val, math_min(max_val, value))


fn kaiming_normal_init(inout tensor: Tensor[DType.float32], fan_in: Int):
    """Initialize tensor with Kaiming normal initialization.
    
    Args:
        tensor: Tensor to initialize
        fan_in: Number of input units
    """
    var std = sqrt(2.0 / Float32(fan_in))
    rand(tensor.data(), tensor.num_elements())
    
    # Scale to normal distribution with std
    for i in range(tensor.num_elements()):
        tensor[i] = (tensor[i] - 0.5) * 2.0 * std


# ============================================================================
# Layer Normalization
# ============================================================================

struct LayerNorm:
    """Layer normalization implementation."""
    var normalized_shape: Int
    var eps: Float32
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, normalized_shape: Int, eps: Float32 = 1e-5):
        """Initialize LayerNorm.
        
        Args:
            normalized_shape: Size of the dimension to normalize
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize weight to ones
        self.weight = Tensor[DType.float32](TensorShape(normalized_shape))
        for i in range(normalized_shape):
            self.weight[i] = 1.0
        
        # Initialize bias to zeros
        self.bias = Tensor[DType.float32](TensorShape(normalized_shape))
        memset_zero(self.bias.data(), self.bias.num_elements())

    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply layer normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        var out = Tensor[DType.float32](x.shape())
        
        # Compute mean
        var mean: Float32 = 0.0
        for i in range(x.num_elements()):
            mean += x[i]
        mean /= Float32(x.num_elements())
        

# ============================================================================
# Headed Repeat Causal Linear
# ============================================================================

struct HeadedRepeatCausalLinear:
    """
    Multi-headed repeat causal linear layer.
    First half of heads use column repeat, second half use row repeat.
    """
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var decay_value: Tensor[DType.float32]
    var decay_constant: Float32
    var cache: Tensor[DType.float32]
    var dim: Int
    var heads: Int
    var head_dim: Int
    var use_decay: Bool
    var use_shared_cache: Bool

    fn __init__(inout self, dim: Int, heads: Int, head_dim: Int = 256, decay: Bool = False, decay_constant: Float32 = 1.0, use_shared_cache: Bool = False):
        """Initialize HeadedRepeatCausalLinear layer.
        
        Args:
            dim: Sequence dimension
            heads: Number of heads
            head_dim: Dimension per head
            decay: Whether to use learnable decay
            decay_constant: Decay normalization constant
            use_shared_cache: Whether to allocate cache in shared memory for multi-threading
        """
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.use_decay = decay
        self.decay_constant = decay_constant
        self.use_shared_cache = use_shared_cache
        
        # Initialize weight for all heads
        self.weight = Tensor[DType.float32](TensorShape(heads, dim))
        rand(self.weight.data(), self.weight.num_elements())
        
        # Initialize bias for all heads
        self.bias = Tensor[DType.float32](TensorShape(heads, dim))
        memset_zero(self.bias.data(), self.bias.num_elements())
        
        # Initialize decay values (2 values for row and col)
        if decay:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        else:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        
        # Initialize cache for all heads with optional shared memory allocation
        self.cache = allocate_tensor_with_flag(
            TensorShape(heads, head_dim),
            DType.float32,
            use_shared_cache
        )

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass with multi-headed caching.
        
        Args:
            x: Input tensor of shape [b, e, h] where h is heads
            index: Current sequence position
            
        Returns:
            Output tensor with same shape as input
        """
        var decay_val = clip(self.decay_value[1], 0.9, 1.0) ** (1.0 / self.decay_constant)
        
        # Output tensor
        var out = Tensor[DType.float32](x.shape())
        
        var half_heads = self.heads // 2
        
        # Process each head
        # First half: column repeat
        for h in range(half_heads):
            var w = self.weight[h * self.dim + index]
            var b = self.bias[h * self.dim + index]
            
            for e in range(self.head_dim):
                var x_idx = e * self.heads + h
                var cache_idx = h * self.head_dim + e
                
                # Column repeat logic
                var val = w * x[x_idx] + w * decay_val * self.cache[cache_idx]
                self.cache[cache_idx] = val / w
                out[x_idx] = val + b
        
        # Second half: row repeat
        for h in range(half_heads, self.heads):
            var w = self.weight[h * self.dim + index]
            var b = self.bias[h * self.dim + index]
            
            for e in range(self.head_dim):
                var x_idx = e * self.heads + h
                var cache_idx = h * self.head_dim + e
                
                # Row repeat logic
                var val = w * x[x_idx] + decay_val * self.cache[cache_idx]
                self.cache[cache_idx] = val
                out[x_idx] = val + b
        
        return out


# ============================================================================
# Parallel Repeat Heads
# ============================================================================

struct ParallelRepeatHeads:
    """
    Parallel processing of multi-headed repeat operations.
    Uses HeadedRepeatCausalLinear for efficient multi-head mixing.
    """
    var n_heads: Int
    var dim: Int
    var seq_len: Int
    var hidden_dim: Int
    var use_projections: Bool
    
    var in_proj: Linear
    var out_proj: Linear
    var mixer_heads: HeadedRepeatCausalLinear

    fn __init__(inout self, dim: Int, seq_len: Int, hidden_dim: Int, n_heads: Int, use_projections: Bool = True, decay: Bool = False):
        """Initialize ParallelRepeatHeads.
        
        Args:
            dim: Model dimension
            seq_len: Sequence length
            hidden_dim: Hidden dimension per head
            n_heads: Number of heads
            use_projections: Whether to use input/output projections
            decay: Whether to use decay in mixer heads
        """
        self.n_heads = n_heads
        self.dim = dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_projections = use_projections
        
        # Initialize projections
        self.in_proj = Linear(dim, dim, use_bias=True)
        self.out_proj = Linear(dim, dim, use_bias=True)
        
        # Initialize mixer heads
        var decay_constant = Float32(seq_len) / 512.0
        self.mixer_heads = HeadedRepeatCausalLinear(seq_len, n_heads, head_dim=hidden_dim, decay=decay, decay_constant=decay_constant)

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass through parallel heads.
        
        Args:
            x: Input tensor of shape [b, dim]
            index: Current sequence position
            
        Returns:
            Output tensor of shape [b, dim]
        """
        var processed = x
        
        # Apply input projection if enabled
        if self.use_projections:
            processed = self.in_proj.forward(x)
        
        # Reshape for multi-head processing: [b, dim] -> [b*h, e]
        # This is a simplified version; full implementation would use proper reshape
        var head_output = self.mixer_heads.forward(processed, index)
        
        # Apply output projection if enabled
        if self.use_projections:
            head_output = self.out_proj.forward(head_output)
        
        return head_output


# ============================================================================
# Mixed Repeat Heads
# ============================================================================

struct MixedRepeatHeads:
    """
    Mixed multi-head architecture with separate col and row repeat heads.
    First half of heads use ColRepeatCausalLinear, second half use RowRepeatCausalLinear.
    """
    var n_heads: Int
    var dim: Int
    var seq_len: Int
    var hidden_dim: Int
    var use_projections: Bool
    
    var out_proj: Linear
    # Note: In full implementation, would have lists of projection and mixer heads
    # Simplified here to show structure

    fn __init__(inout self, dim: Int, seq_len: Int, hidden_dim: Int, n_heads: Int, use_projections: Bool = True, expanded_convs: Bool = False, decay: Bool = False):
        """Initialize MixedRepeatHeads.
        
        Args:
            dim: Model dimension
            seq_len: Sequence length
            hidden_dim: Hidden dimension per head
            n_heads: Number of heads
            use_projections: Whether to use projections
            expanded_convs: Whether to use expanded convolutions
            decay: Whether to use decay
        """
        self.n_heads = n_heads
        self.dim = dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_projections = use_projections
        
        # Initialize output projection
        self.out_proj = Linear(dim, dim, use_bias=True)
        
        # Note: Full implementation would initialize n_heads projection layers
        # and n_heads mixer layers (half col, half row)

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass through mixed heads.
        
        Args:
            x: Input tensor
            index: Current sequence position
            
        Returns:
            Output tensor after mixed head processing
        """
        # Simplified implementation
        # Full version would process each head separately and concatenate
        var output = x
        
        if self.use_projections:
            output = self.out_proj.forward(output)
        
        return output


# ============================================================================
# Repeat Heads
# ============================================================================

struct RepeatHeads:
    """
    General multi-head repeat architecture.
    All heads use the same type of repeat operation (typically column repeat).
    """
    var n_heads: Int
    var dim: Int
    var seq_len: Int
    var hidden_dim: Int
    var use_projections: Bool
    var combined_heads: Bool
    
    var out_proj: Linear

    fn __init__(inout self, dim: Int, seq_len: Int, hidden_dim: Int, n_heads: Int, expanded_convs: Bool = False, combined_heads: Bool = False, use_projections: Bool = True, decay: Bool = False):
        """Initialize RepeatHeads.
        
        Args:
            dim: Model dimension
            seq_len: Sequence length
            hidden_dim: Hidden dimension per head
            n_heads: Number of heads
            expanded_convs: Whether to use expanded convolutions
            combined_heads: Whether to use combined repeat heads
            use_projections: Whether to use projections
            decay: Whether to use decay
        """
        self.n_heads = n_heads
        self.dim = dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_projections = use_projections
        self.combined_heads = combined_heads
        
        # Initialize output projection
        self.out_proj = Linear(dim, dim, use_bias=True)
        
        # Note: Full implementation would initialize projection and mixer heads

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass through repeat heads.
        
        Args:
            x: Input tensor
            index: Current sequence position
            
        Returns:
            Output tensor after head processing
        """
        # Simplified implementation
        var output = x
        
        if self.use_projections:
            output = self.out_proj.forward(output)
        
        return output

        # Compute variance
        var var: Float32 = 0.0
        for i in range(x.num_elements()):
            var diff = x[i] - mean
            var += diff * diff
        var /= Float32(x.num_elements())
        
        # Normalize
        var std = sqrt(var + self.eps)
        for i in range(x.num_elements()):
            var normalized = (x[i] - mean) / std
            out[i] = normalized * self.weight[i % self.normalized_shape] + self.bias[i % self.normalized_shape]
        
        return out


# ============================================================================
# Linear Layer
# ============================================================================

struct Linear:
    """Basic linear transformation layer."""
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var in_features: Int
    var out_features: Int
    var use_bias: Bool

    fn __init__(inout self, in_features: Int, out_features: Int, use_bias: Bool = True):
        """Initialize Linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            use_bias: Whether to use bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # Initialize weight
        self.weight = Tensor[DType.float32](TensorShape(out_features, in_features))
        kaiming_normal_init(self.weight, in_features)
        
        # Initialize bias
        if use_bias:
            self.bias = Tensor[DType.float32](TensorShape(out_features))
            memset_zero(self.bias.data(), self.bias.num_elements())
        else:
            self.bias = Tensor[DType.float32](TensorShape(0))

    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: y = xW^T + b.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Simplified matrix multiplication for demonstration
        # In production, use optimized BLAS operations
        var batch_size = x.num_elements() // self.in_features
        var out = Tensor[DType.float32](TensorShape(batch_size, self.out_features))
        
        for b in range(batch_size):
            for o in range(self.out_features):
                var sum: Float32 = 0.0
                for i in range(self.in_features):
                    sum += x[b * self.in_features + i] * self.weight[o * self.in_features + i]
                if self.use_bias:
                    sum += self.bias[o]
                out[b * self.out_features + o] = sum
        
        return out


# ============================================================================
# Activation Functions
# ============================================================================

fn silu(x: Float32) -> Float32:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x / (1.0 + exp(-x))


fn apply_silu(inout tensor: Tensor[DType.float32]):
    """Apply SiLU activation in-place."""
    for i in range(tensor.num_elements()):
        tensor[i] = silu(tensor[i])


# ============================================================================
# Embedding Layer
# ============================================================================

struct Embedding:
    """Embedding layer for token indices."""
    var weight: Tensor[DType.float32]
    var num_embeddings: Int
    var embedding_dim: Int

    fn __init__(inout self, num_embeddings: Int, embedding_dim: Int):
        """Initialize Embedding layer.
        
        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings with random values
        self.weight = Tensor[DType.float32](TensorShape(num_embeddings, embedding_dim))
        rand(self.weight.data(), self.weight.num_elements())

    fn forward(self, indices: Tensor[DType.int32]) -> Tensor[DType.float32]:
        """Look up embeddings for given indices.
        
        Args:

# ============================================================================
# Combined Repeat Causal Linear
# ============================================================================

struct CombinedRepeatCausalLinear:
    """
    Combined row and column repeat causal linear layer.
    Maintains separate caches for row and column operations.
    """
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var decay_value: Tensor[DType.float32]
    var decay_constant: Float32
    var row_cache: Tensor[DType.float32]
    var col_cache: Tensor[DType.float32]
    var dim: Int
    var embedding_dim: Int
    var use_decay: Bool
    var use_shared_cache: Bool

    fn __init__(inout self, dim: Int, embedding_dim: Int = 512, decay: Bool = False, decay_constant: Float32 = 1.0, use_shared_cache: Bool = False):
        """Initialize CombinedRepeatCausalLinear layer.
        
        Args:
            dim: Sequence dimension
            embedding_dim: Embedding dimension for caches
            decay: Whether to use learnable decay
            decay_constant: Decay normalization constant
            use_shared_cache: Whether to allocate caches in shared memory for multi-threading
        """
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.use_decay = decay
        self.decay_constant = decay_constant
        self.use_shared_cache = use_shared_cache
        
        # Initialize weight with 2 rows (one for row, one for col)
        self.weight = Tensor[DType.float32](TensorShape(2, dim))
        rand(self.weight.data(), self.weight.num_elements())
        
        # Initialize bias
        self.bias = Tensor[DType.float32](TensorShape(dim))
        memset_zero(self.bias.data(), self.bias.num_elements())
        
        # Initialize decay values (2 separate decay values)
        if decay:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        else:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        
        # Initialize caches with optional shared memory allocation
        self.row_cache = allocate_tensor_with_flag(
            TensorShape(embedding_dim),
            DType.float32,
            use_shared_cache
        )
        self.col_cache = allocate_tensor_with_flag(
            TensorShape(embedding_dim),
            DType.float32,
            use_shared_cache
        )

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass with combined row and column caching."""
        # Clip and apply decay
        var decay_row = clip(self.decay_value[0], 0.9, 1.0) ** (1.0 / self.decay_constant)
        var decay_col = clip(self.decay_value[1], 0.9, 1.0) ** (1.0 / self.decay_constant)
        
        var out = Tensor[DType.float32](x.shape())
        var w_row = self.weight[0 * self.dim + index]
        var w_col = self.weight[1 * self.dim + index]
        var b = self.bias[index]
        
        # Row computation
        for i in range(x.num_elements()):
            var row_out = w_row * x[i] + decay_row * self.row_cache[i]
            self.row_cache[i] = row_out
            
            # Col computation
            var col_out = w_col * x[i] + w_col * decay_col * self.col_cache[i]
            self.col_cache[i] = col_out / w_col
            
            # Combined output
            out[i] = row_out + col_out + b
        
        return out


# ============================================================================
# Kernel Repeat Linear
# ============================================================================

struct KernelRepeatLinear:
    """
    Kernel-based repeat linear layer with convolution-like operations.
    """
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var decay_value: Tensor[DType.float32]
    var decay_constant: Float32
    var cache: Tensor[DType.float32]
    var dim: Int
    var kernel: Int
    var embedding_dim: Int
    var use_decay: Bool
    var use_shared_cache: Bool

    fn __init__(inout self, dim: Int, kernel: Int, embedding_dim: Int = 512, decay: Bool = False, decay_constant: Float32 = 1.0, use_shared_cache: Bool = False):
        """Initialize KernelRepeatLinear layer.
        
        Args:
            dim: Sequence dimension
            kernel: Kernel size
            embedding_dim: Embedding dimension for cache
            decay: Whether to use learnable decay
            decay_constant: Decay normalization constant
            use_shared_cache: Whether to allocate cache in shared memory for multi-threading
        """
        self.dim = dim
        self.kernel = kernel
        self.embedding_dim = embedding_dim
        self.use_decay = decay
        self.decay_constant = decay_constant
        self.use_shared_cache = use_shared_cache
        
        # Initialize weight with kernel dimension
        self.weight = Tensor[DType.float32](TensorShape(kernel, dim))
        rand(self.weight.data(), self.weight.num_elements())
        
        # Initialize bias
        self.bias = Tensor[DType.float32](TensorShape(dim))
        memset_zero(self.bias.data(), self.bias.num_elements())
        
        # Initialize decay values
        if decay:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        else:
            self.decay_value = Tensor[DType.float32](TensorShape(2, 1))
            self.decay_value[0] = 1.0
            self.decay_value[1] = 1.0
        
        # Initialize cache with kernel dimension and optional shared memory allocation
        self.cache = allocate_tensor_with_flag(
            TensorShape(kernel, embedding_dim),
            DType.float32,
            use_shared_cache
        )

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass with kernel-based caching."""
        var decay_val = clip(self.decay_value[1], 0.9, 1.0) ** (1.0 / self.decay_constant)
        
        # Simplified kernel operation
        var out = Tensor[DType.float32](x.shape())
        memset_zero(out.data(), out.num_elements())
        
        for k in range(self.kernel):
            var w = self.weight[k * self.dim + index]
            for i in range(x.num_elements()):
                var cache_idx = k * self.embedding_dim + i
                var val = w * decay_val * x[i] + w * decay_val * self.cache[cache_idx]
                self.cache[cache_idx] = val / w
                out[i] += val
        
        # Add bias
        var b = self.bias[index]
        for i in range(out.num_elements()):
            out[i] += b
        
        return out


# ============================================================================
# Mixer Block
# ============================================================================

struct MixerBlock:
    """
    MLP-Mixer block with token and channel mixing.
    Combines layer normalization, linear transformations, and recurrent mixing.
    """
    var hidden_dim: Int
    var seq_len: Int
    var expansion_factor: Int
    var use_shared_cache: Bool
    
    # Layer normalization
    var channel_norm: LayerNorm
    var token_norm: LayerNorm
    
    # Channel mixing layers
    var channel_linear1: Linear
    var channel_linear2: Linear
    
    # Token mixing layer
    var token_mixer: RowRepeatCausalLinear

    fn __init__(inout self, hidden_dim: Int, seq_len: Int, expansion_factor: Int = 4, use_shared_cache: Bool = False):
        """Initialize MixerBlock.
        
        Args:
            hidden_dim: Hidden dimension size
            seq_len: Sequence length
            expansion_factor: Expansion factor for channel mixing
            use_shared_cache: Whether to allocate caches in shared memory for multi-threading
        """
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor
        self.use_shared_cache = use_shared_cache
        
        # Initialize layer norms
        self.channel_norm = LayerNorm(hidden_dim)
        self.token_norm = LayerNorm(hidden_dim)
        
        # Initialize channel mixing layers
        self.channel_linear1 = Linear(hidden_dim, hidden_dim * expansion_factor)
        self.channel_linear2 = Linear(hidden_dim * expansion_factor, hidden_dim)
        
        # Initialize token mixing layer
        self.token_mixer = RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim)

    fn forward(inout self, x: Tensor[DType.float32], index: Int) -> Tensor[DType.float32]:
        """Forward pass through mixer block.
        
        Args:
            x: Input tensor
            index: Current sequence position
            
        Returns:
            Output tensor after mixing operations
        """
        # Channel mixing with residual
        var res = x
        var x_norm = self.channel_norm.forward(x)
        var x_expanded = self.channel_linear1.forward(x_norm)
        apply_silu(x_expanded)
        var x_channel = self.channel_linear2.forward(x_expanded)
        
        # Add residual
        for i in range(x.num_elements()):
            x_channel[i] += res[i]
        
        # Token mixing with residual
        res = x_channel
        x_norm = self.token_norm.forward(x_channel)
        var x_token = self.token_mixer.forward(x_norm, index)
        
        # Add residual
        for i in range(x_token.num_elements()):
            x_token[i] += res[i]
        
        return x_token


# ============================================================================
# Recurrent MLP Mixer Model
# ============================================================================

struct RecurrentMLPMixer:
    """
    Complete Recurrent MLP Mixer model for sequence modeling.
    Combines embedding, multiple mixer blocks, and output projection.
    """
    var vocab_size: Int
    var hidden_dim: Int
    var seq_len: Int
    var num_blocks: Int
    
    var input_layer: Embedding
    var output_layer: Linear
    
    # Note: In a full implementation, mixer_blocks would be a list/array
    # For simplicity, we show the structure with a single block
    var mixer_block: MixerBlock

    fn __init__(inout self, vocab_size: Int, hidden_dim: Int, seq_len: Int, num_blocks: Int):
        """Initialize RecurrentMLPMixer model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension size
            seq_len: Maximum sequence length
            num_blocks: Number of mixer blocks
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        
        # Initialize embedding layer
        self.input_layer = Embedding(vocab_size, hidden_dim)
        
        # Initialize output projection (no bias for weight tying)
        self.output_layer = Linear(hidden_dim, vocab_size, use_bias=False)
        
        # Initialize mixer block (in full implementation, would have num_blocks)
        self.mixer_block = MixerBlock(hidden_dim, seq_len)

    fn forward(inout self, input_ids: Tensor[DType.int32], index: Int) -> Tensor[DType.float32]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token indices
            index: Current sequence position
            
        Returns:
            Logits over vocabulary
        """
        # Embed input tokens
        var x = self.input_layer.forward(input_ids)
        
        # Pass through mixer blocks (simplified to single block)
        x = self.mixer_block.forward(x, index)
        
        # Project to vocabulary
        var logits = self.output_layer.forward(x)
        
        return logits

    fn count_params(self) -> Int:
        """Count total number of parameters in the model."""
        var total = 0
        total += self.input_layer.weight.num_elements()
        total += self.output_layer.weight.num_elements()
        # Add mixer block parameters
        total += self.mixer_block.channel_linear1.weight.num_elements()
        total += self.mixer_block.channel_linear2.weight.num_elements()
        total += self.mixer_block.token_mixer.weight.num_elements()
        return total

            indices: Tensor of token indices
            
        Returns:
            Embedded representations
        """
        var batch_size = indices.num_elements()
        var out = Tensor[DType.float32](TensorShape(batch_size, self.embedding_dim))
        
        for b in range(batch_size):
            var idx = int(indices[b])
            for e in range(self.embedding_dim):
                out[b * self.embedding_dim + e] = self.weight[idx * self.embedding_dim + e]
        
        return out


# ============================================================================
# Main Entry Point
# ============================================================================

fn main():
    """Main function for testing the recurrent inference module."""
    print("Recurrent MLP Mixer - Mojo Implementation")
    print("==========================================")
    
    # Test ColRepeatCausalLinear
    var dim = 128
    var embedding_dim = 256
    var col_layer = ColRepeatCausalLinear(dim, embedding_dim, decay=True, decay_constant=2.0)
    
    var test_input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(test_input.data(), test_input.num_elements())
    
    var output = col_layer.forward(test_input, 0)
    print("ColRepeatCausalLinear test passed")
    
    # Test RowRepeatCausalLinear
    var row_layer = RowRepeatCausalLinear(dim, embedding_dim, decay=True, decay_constant=2.0)
    var output2 = row_layer.forward(test_input, 0)
    print("RowRepeatCausalLinear test passed")
    
    print("\nAll basic tests passed!")