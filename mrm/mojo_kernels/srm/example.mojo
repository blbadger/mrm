"""
Example usage of the Recurrent MLP Mixer in Mojo

This file demonstrates how to use the various components of the
Structured Recurrent Mixer (SRM) implementation.
"""

from tensor import Tensor, TensorShape
from random import rand
from recurrent_inference import (
    ColRepeatCausalLinear,
    RowRepeatCausalLinear,
    CombinedRepeatCausalLinear,
    KernelRepeatLinear,
    MixerBlock,
    RecurrentMLPMixer,
    LayerNorm,
    Linear,
    Embedding,
)


fn example_col_repeat_layer():
    """Demonstrate ColRepeatCausalLinear usage."""
    print("\n=== ColRepeatCausalLinear Example ===")
    
    var dim = 128
    var embedding_dim = 256
    var layer = ColRepeatCausalLinear(dim, embedding_dim, decay=True, decay_constant=2.0)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass at different sequence positions
    for i in range(5):
        var output = layer.forward(input, i)
        print("Position", i, "- Output shape:", output.shape().__str__())
    
    print("✓ ColRepeatCausalLinear example completed")


fn example_col_repeat_layer_shared_cache():
    """Demonstrate ColRepeatCausalLinear with shared memory cache."""
    print("\n=== ColRepeatCausalLinear with Shared Cache Example ===")
    
    var dim = 128
    var embedding_dim = 256
    var layer = ColRepeatCausalLinear(
        dim,
        embedding_dim,
        decay=True,
        decay_constant=2.0,
        use_shared_cache=True  # Enable shared memory cache
    )
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass
    var output = layer.forward(input, 0)
    print("Output shape:", output.shape().__str__())
    print("Cache allocated in shared memory for multi-threaded access")
    print("✓ ColRepeatCausalLinear with shared cache example completed")


fn example_row_repeat_layer():
    """Demonstrate RowRepeatCausalLinear usage."""
    print("\n=== RowRepeatCausalLinear Example ===")
    
    var dim = 128
    var embedding_dim = 256
    var layer = RowRepeatCausalLinear(dim, embedding_dim, decay=True, decay_constant=2.0)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass
    var output = layer.forward(input, 0)
    print("Output shape:", output.shape().__str__())
    print("✓ RowRepeatCausalLinear example completed")


fn example_combined_layer():
    """Demonstrate CombinedRepeatCausalLinear usage."""
    print("\n=== CombinedRepeatCausalLinear Example ===")
    
    var dim = 128
    var embedding_dim = 512
    var layer = CombinedRepeatCausalLinear(dim, embedding_dim, decay=True, decay_constant=2.0)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass
    var output = layer.forward(input, 0)
    print("Output shape:", output.shape().__str__())
    print("✓ CombinedRepeatCausalLinear example completed")


fn example_kernel_layer():
    """Demonstrate KernelRepeatLinear usage."""
    print("\n=== KernelRepeatLinear Example ===")
    
    var dim = 128
    var kernel = 3
    var embedding_dim = 512
    var layer = KernelRepeatLinear(dim, kernel, embedding_dim, decay=True, decay_constant=2.0)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(embedding_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass
    var output = layer.forward(input, 0)
    print("Kernel size:", kernel)
    print("Output shape:", output.shape().__str__())
    print("✓ KernelRepeatLinear example completed")


fn example_mixer_block():
    """Demonstrate MixerBlock usage."""
    print("\n=== MixerBlock Example ===")
    
    var hidden_dim = 256
    var seq_len = 128
    var expansion_factor = 4
    
    var block = MixerBlock(hidden_dim, seq_len, expansion_factor)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(hidden_dim))
    rand(input.data(), input.num_elements())
    
    # Forward pass through mixer block
    var output = block.forward(input, 0)
    print("Hidden dim:", hidden_dim)
    print("Sequence length:", seq_len)
    print("Expansion factor:", expansion_factor)
    print("Output shape:", output.shape().__str__())
    print("✓ MixerBlock example completed")


fn example_full_model():
    """Demonstrate full RecurrentMLPMixer model."""
    print("\n=== RecurrentMLPMixer Model Example ===")
    
    # Model configuration (smaller for example)
    var vocab_size = 1000
    var hidden_dim = 256
    var seq_len = 128
    var num_blocks = 4
    
    var model = RecurrentMLPMixer(vocab_size, hidden_dim, seq_len, num_blocks)
    
    # Create input token indices
    var batch_size = 8
    var input_ids = Tensor[DType.int32](TensorShape(batch_size))
    for i in range(batch_size):
        input_ids[i] = i % vocab_size
    
    # Forward pass
    var logits = model.forward(input_ids, 0)
    
    print("Model Configuration:")
    print("  Vocabulary size:", vocab_size)
    print("  Hidden dimension:", hidden_dim)
    print("  Sequence length:", seq_len)
    print("  Number of blocks:", num_blocks)
    print("  Batch size:", batch_size)
    print("\nOutput:")
    print("  Logits shape:", logits.shape().__str__())
    print("  Total parameters:", model.count_params())
    print("✓ Full model example completed")


fn example_full_model_shared_cache():
    """Demonstrate full RecurrentMLPMixer model with shared memory caches."""
    print("\n=== RecurrentMLPMixer Model with Shared Cache Example ===")
    
    # Model configuration (smaller for example)
    var vocab_size = 1000
    var hidden_dim = 256
    var seq_len = 128
    var num_blocks = 4
    
    # Initialize model with shared memory caches
    var model = RecurrentMLPMixer(
        vocab_size,
        hidden_dim,
        seq_len,
        num_blocks,
        use_shared_cache=True  # All caches in shared memory
    )
    
    # Create input token indices
    var batch_size = 8
    var input_ids = Tensor[DType.int32](TensorShape(batch_size))
    for i in range(batch_size):
        input_ids[i] = i % vocab_size
    
    # Forward pass
    var logits = model.forward(input_ids, 0)
    
    print("Model Configuration:")
    print("  Vocabulary size:", vocab_size)
    print("  Hidden dimension:", hidden_dim)
    print("  Sequence length:", seq_len)
    print("  Number of blocks:", num_blocks)
    print("  Batch size:", batch_size)
    print("  Shared cache: ENABLED")
    print("\nOutput:")
    print("  Logits shape:", logits.shape().__str__())
    print("  Total parameters:", model.count_params())
    print("  All internal caches allocated in shared memory")
    print("✓ Full model with shared cache example completed")


fn example_layer_norm():
    """Demonstrate LayerNorm usage."""
    print("\n=== LayerNorm Example ===")
    
    var normalized_shape = 256
    var layer_norm = LayerNorm(normalized_shape)
    
    # Create random input
    var input = Tensor[DType.float32](TensorShape(normalized_shape))
    rand(input.data(), input.num_elements())
    
    # Apply normalization
    var output = layer_norm.forward(input)
    print("Normalized shape:", normalized_shape)
    print("Output shape:", output.shape().__str__())
    print("✓ LayerNorm example completed")


fn example_linear_layer():
    """Demonstrate Linear layer usage."""
    print("\n=== Linear Layer Example ===")
    
    var in_features = 256
    var out_features = 512
    var linear = Linear(in_features, out_features, use_bias=True)
    
    # Create random input
    var batch_size = 4
    var input = Tensor[DType.float32](TensorShape(batch_size, in_features))
    rand(input.data(), input.num_elements())
    
    # Forward pass
    var output = linear.forward(input)
    print("Input features:", in_features)
    print("Output features:", out_features)
    print("Batch size:", batch_size)
    print("Output shape:", output.shape().__str__())
    print("✓ Linear layer example completed")


fn example_embedding():
    """Demonstrate Embedding layer usage."""
    print("\n=== Embedding Layer Example ===")
    
    var num_embeddings = 1000
    var embedding_dim = 256
    var embedding = Embedding(num_embeddings, embedding_dim)
    
    # Create token indices
    var batch_size = 8
    var indices = Tensor[DType.int32](TensorShape(batch_size))
    for i in range(batch_size):
        indices[i] = i * 10
    
    # Look up embeddings
    var output = embedding.forward(indices)
    print("Vocabulary size:", num_embeddings)
    print("Embedding dimension:", embedding_dim)
    print("Batch size:", batch_size)
    print("Output shape:", output.shape().__str__())
    print("✓ Embedding layer example completed")


fn main():
    """Run all examples."""
    print("=" * 60)
    print("Recurrent MLP Mixer - Mojo Examples")
    print("=" * 60)
    
    # Run individual component examples
    example_col_repeat_layer()
    example_col_repeat_layer_shared_cache()
    example_row_repeat_layer()
    example_combined_layer()
    example_kernel_layer()
    example_layer_norm()
    example_linear_layer()
    example_embedding()
    example_mixer_block()
    example_full_model()
    example_full_model_shared_cache()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)