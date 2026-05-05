from __future__ import annotations

from collections.abc import Callable
from typing import Any
import asyncio

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
import traceback

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')


class ColRepeatCausalLinear(Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        self.weight = Tensor.zeros([1, dim])
        self.bias = Tensor.zeros([dim])
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self._index = 0
        # cache removed — now passed explicitly as a graph tensor

    # ColRepeatCausalLinear.forward
    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        idx = self._index
        decay_value = (self.decay_value.clip(min=0.9, max=1) ** (1 / self.decay_constant)).to(x.device)
        w = self.weight[0:1, idx:idx+1].to(x.device)
        b = self.bias[idx:idx+1].to(x.device)
        out = w * x + w * decay_value * cache + b
        new_cache = (out - b) / w
        return out, new_cache

class RowRepeatCausalLinear(Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        self.weight = Tensor.ones([1, dim])
        self.bias = Tensor.zeros([dim])
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self._index = 0
        # cache removed — now passed explicitly as a graph tensor

    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        idx = self._index
        decay_value = (self.decay_value.clip(min=0.9, max=1) ** (1 / self.decay_constant)).to(x.device)
        w = self.weight[0:1, idx:idx+1].to(x.device)
        b = self.bias[idx:idx+1].to(x.device)
        out = w * x + decay_value * cache + b
        new_cache = out - b
        return out, new_cache


class HeadedRepeatCausalLinear(nn.Module):
    """
    Mixed-headed repeat module for ParallelRepeatHeads
    """
    def __init__(self, dim: int, heads: int, head_dim=256, decay=False, decay_constant=1):
        super().__init__()
        self.weight = Tensor.ones([heads, dim])
        self.bias = Tensor.zeros([heads, dim])
        self.heads = heads
        self.decay_value = Tensor.ones([2, 1])  # N.B. only one value used, for back compatibility
        self.decay_constant = decay_constant
        self.head_dim = head_dim
        # cache removed — now passed explicitly as a graph tensor

    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        # x shape: (b*h, e)   cache shape: (b, h, head_dim)
        idx = self._index
        x = x.reshape([x.shape[0] // self.heads, x.shape[1], self.heads])  # (b, e, h)
        decay_value = (self.decay_value.clip(min=0.9, max=1) ** (1 / self.decay_constant))
        cache_beh = cache.permute([0, 2, 1])  # (b, e, h)

        # static slices of weight/bias at trace time
        w = self.weight[:, idx:idx+1]   # (heads, 1)
        b = self.bias[:, idx:idx+1]     # (heads, 1)

        # row computation — second half of heads
        row_out = (w[self.heads // 2:] * x[..., self.heads // 2:]
                   + decay_value[1] * cache_beh[..., self.heads // 2:])
        row_cache = row_out

        # col computation — first half of heads
        col_out = (w[:self.heads // 2] * x[..., :self.heads // 2]
                   + w[:self.heads // 2] * decay_value[1] * cache_beh[..., :self.heads // 2])
        col_cache = col_out / w[:self.heads // 2]

        new_cache = F.concat([row_cache, col_cache], axis=-1).permute([0, 2, 1])  # (b, h, e)

        output = F.concat([col_out, row_out], axis=-1) + b
        output = output.reshape([x.shape[0] * self.heads, x.shape[1]])
        return output, new_cache


class ParallelRepeatHeads(Module):

    def __init__(
        self,
        dim: int,
        seq_len: int,
        head_dim: int,
        n_heads: int,
        use_projections=True,
        decay=False,
        **kwargs
    ):
        print('parallel heads')
        self.n_heads = n_heads
        self.head_dim = head_dim
        if use_projections:
            self.in_proj = max.nn.Linear(dim, dim)
            self.out_proj = max.nn.Linear(dim, dim)
        self.mixer_heads = HeadedRepeatCausalLinear(
            seq_len, n_heads, head_dim=head_dim, decay=decay, decay_constant=seq_len // 512
        )
        self.use_projections = use_projections

    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        # cache shape: (b, n_heads, head_dim) — one tensor for the whole block
        batch_dim = x.shape[0]
        if self.use_projections:
            x = self.in_proj(x)
        projections = x.reshape([batch_dim * self.n_heads, self.head_dim])
        conv_projection, new_cache = self.mixer_heads(projections, cache)
        output = conv_projection.reshape([batch_dim, self.n_heads * self.head_dim])
        if self.use_projections:
            output = self.out_proj(output)
        return output, new_cache


class MixedRepeatHeads(Module):

    def __init__(self, dim: int, seq_len: int, hidden_dim: int, n_heads: int,
                 expanded_convs=False, decay=False, use_projections=True):
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.use_projections = use_projections
        if use_projections:
            self.proj_head = max.nn.sequential.ModuleList(
                max.nn.Linear(dim, hidden_dim) for _ in range(n_heads)
            )
            self.out_proj = max.nn.Linear(dim, dim)

        self.mixer_heads = (
            max.nn.sequential.ModuleList(
                ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len // 512)
                for _ in range(n_heads // 2)
            )
            + max.nn.sequential.ModuleList(
                RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len // 512)
                for _ in range(n_heads // 2)
            )
        )

    def forward(self, x: Tensor, caches: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        # caches: list of n_heads tensors, each (b, hidden_dim)
        activations = []
        new_caches = []
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[:, head * self.hidden_dim:(head + 1) * self.hidden_dim]
            out, new_cache = self.mixer_heads[head](projection, caches[head])
            activations.append(out)
            new_caches.append(new_cache)

        hidden_layer = F.concat(activations, axis=1)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)
        return hidden_layer, new_caches


class LayerNorm(Module):

    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class MixerBlock(Module):

    def __init__(self, hidden_dim: int,
        seq_len: int,
        expansion_factor=4,
        heads=None,
        mixed_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True
    ):
        print('Block initializing...')
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor
        self.parallel_heads = parallel_heads
        self.mixed_heads = mixed_heads

        self.channel_norm = LayerNorm(hidden_dim)
        self.token_norm = LayerNorm(hidden_dim)

        self.channel_in = max.nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.channel_out = max.nn.Linear(hidden_dim * expansion_factor, hidden_dim)

        if heads is not None and heads > 0:
            if parallel_heads:
                self.token_mixing_layer = ParallelRepeatHeads(
                    hidden_dim, seq_len, hidden_dim // heads, heads,
                    use_projections=use_projections, decay=decay
                )
            elif mixed_heads:
                self.token_mixing_layer = MixedRepeatHeads(
                    hidden_dim, seq_len, hidden_dim // heads, heads,
                    use_projections=use_projections, decay=decay
                )
            else:
                self.token_mixing_layer = RepeatHeads(
                    hidden_dim, seq_len, hidden_dim // heads, heads, decay=decay
                )
        else:
            self.token_mixing_layer = ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim)

    def forward(self, x: Tensor, caches: list[Tensor] | Tensor) -> tuple[Tensor, list[Tensor] | Tensor]:
        # channel mixing
        res = x
        x = self.channel_norm(x)
        x = self.channel_in(x)
        x = F.silu(x)
        x = self.channel_out(x)
        x = x + res

        # token mixing — caches threaded through
        res = x
        x = self.token_norm(x)
        x, new_caches = self.token_mixing_layer(x, caches)
        x = x + res
        return x, new_caches


class RecurrentSRM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        copy=False,
        mixed_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.heads = heads
        self.parallel_heads = parallel_heads
        self.mixed_heads = mixed_heads

        self.input_layer = max.nn.Embedding(vocab_size, dim=hidden_dim)
        self.mixer_blocks = nn.sequential.ModuleList(
            MixerBlock(
                hidden_dim, seq_len, heads=heads, mixed_heads=mixed_heads,
                decay=decay, parallel_heads=parallel_heads, use_projections=use_projections
            ) for _ in range(num_blocks)
        )
        self.output_layer = max.nn.Linear(hidden_dim, vocab_size, bias=False)

    def _set_index(self, index: int) -> None:
        """Propagate the current sequence position to every mixer head as a plain Python int."""
        for block in self.mixer_blocks:
            for head in block.token_mixing_layer.mixer_heads:
                head._index = index

    def forward(self, input_ids: Tensor, *flat_caches: Tensor) -> tuple[Tensor, ...]:
        n_heads = self.heads if self.heads is not None else 1
        caches_per_block = n_heads if self.mixed_heads else 1

        block_caches = [
            list(flat_caches[i * caches_per_block:(i + 1) * caches_per_block])
            if caches_per_block > 1
            else flat_caches[i]
            for i in range(self.num_blocks)
        ]

        input_ids = input_ids[:, -1]
        x = self.input_layer(input_ids)

        new_block_caches = []
        for i, block in enumerate(self.mixer_blocks):
            x, new_caches = block(x, block_caches[i])
            new_block_caches.append(new_caches)

        output = self.output_layer(x)

        flat_new_caches = []
        flat_new_caches = [c for bc in new_block_caches for c in bc]

        return output, *flat_new_caches


if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)

    dtype, device = defaults()
    print(f'Device: {device}')
    input_string = 'Four score and seven years ago, our forefathers, for the purpose of creating'
    input_tokens = tokenizer(input_string, return_tensors='pt').input_ids[:, 1].unsqueeze(1).to(torch.int64)
    batch_size = 100
    input_tokens = input_tokens.repeat(batch_size, 1)

    tokenized_length = 512
    dim = 128
    layers = 1
    n_heads = 4
    head_dim = dim // n_heads  # 32
    generate_steps = 20

    with default_device(CPU()), default_dtype(dtype):
        model = RecurrentSRM(
            n_vocab,
            dim,
            tokenized_length,
            layers,
            heads=n_heads,
            copy=False,
            mixed_heads=True,
            decay=True,
            parallel_heads=False,
            use_projections=True
        )

    int_index = input_tokens.shape[-1]
    print('model initialized')
    model = model.to(device)
    print('model on device')

    token_type = TensorType(
        DType.int64, shape=[input_tokens.shape[0], 1],
        device=DeviceRef.from_device(device)
    )

    # MixedRepeatHeads: one cache tensor per head per block, shape (batch, head_dim)
    head_cache_type = TensorType(
        dtype, shape=[batch_size, head_dim],
        device=DeviceRef.from_device(device)
    )
    cache_types = [head_cache_type] * (n_heads * layers)

    print(f'Compiling {generate_steps} position graphs...')
    compile_start = time.time()
    compiled_steps: dict[int, object] = {}
    for pos in range(int_index, int_index + generate_steps):
        print (pos)
        model._set_index(pos)
        print ('pos set')
        compiled_steps[pos] = model.compile(token_type, *cache_types)
        print (f'{pos} compiled')

    print(f'Compilation completed in {compile_end - compile_start:.1f}s')
    input_tensor = Tensor.constant(input_tokens[:, -1:], dtype=DType.int64, device=device)

    # Initialise flat caches to zero
    flat_caches = [
        Tensor.zeros([batch_size, head_dim], dtype=dtype, device=device)
        for _ in range(n_heads * layers)
    ]

    start = time.time()
    for i in range(generate_steps):
        pos = int_index + i
        logits, *flat_caches = compiled_steps[pos](input_tensor, *flat_caches)
        input_tensor = F.argmax(logits, axis=1).unsqueeze(1)

    end = time.time()
    print(f"etime: {end - start}")
    total_tokens = generate_steps * batch_size
    print(f'{total_tokens / (end - start):.1f} tokens per second')
    print(f'output shape: {logits.shape}')