"""
SRM (Structured Recurrent Mixer) Package
======================================

This package contains Mojo implementations of recurrent MLP mixer architectures
ported from the PyTorch implementation in mrm/mrm/recurrent_inference.py.

Main Components:
- ColRepeatCausalLinear: Column-repeat causal linear layer
- RowRepeatCausalLinear: Row-repeat causal linear layer
- CombinedRepeatCausalLinear: Combined row/column repeat layer
- KernelRepeatLinear: Kernel-based repeat layer
- MixerBlock: Complete mixer block with token and channel mixing
- RecurrentMLPMixer: Full recurrent MLP mixer model

Usage:
    from mojo_kernels.srm.recurrent_inference import RecurrentMLPMixer
    
    var model = RecurrentMLPMixer(
        vocab_size=8000,
        hidden_dim=1024,
        seq_len=1024,
        num_blocks=16
    )
"""

from .recurrent_inference import (
    ColRepeatCausalLinear,
    RowRepeatCausalLinear,
    CombinedRepeatCausalLinear,
    KernelRepeatLinear,
    HeadedRepeatCausalLinear,
    ParallelRepeatHeads,
    MixedRepeatHeads,
    RepeatHeads,
    MixerBlock,
    RecurrentMLPMixer,
    LayerNorm,
    Linear,
    Embedding,
)