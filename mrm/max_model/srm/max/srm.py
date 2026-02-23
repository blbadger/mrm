from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.tensor import Tensor, TensorType, defaults
import max.functional as F
import max
import max.nn as nn
from max import driver
import torch
from transformers import AutoTokenizer

import os 
from dotenv import load_dotenv
import pathlib
import time

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')


class ColRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        # Standard weight + bias
        self.weight = Tensor.zeros([1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self.cache = torch.zeros([embedding_dim]) # TODO: initialize and send to shared mem via custom op

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        decay_value = (self.decay_value.clip( min=0.9, max=1)**(1/self.decay_constant))
        out = self.weight[0, index]*x + self.weight[0, index]*decay_value*self.cache + self.bias[index]
        self.cache = (out - self.bias[index]) / self.weight[0, index] # cache update: factor out weight, remove bias
        return out

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Column Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"

class RowRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        # Standard weight + bias
        self.weight = Tensor.ones([1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self.cache = torch.zeros([embedding_dim]) # TODO: initialize and send to shared mem via custom op

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        # expects x in shape [B, E]
        decay_value = (self.decay_value.clip(min=0.9, max=1)**(1/self.decay_constant))
        out = self.weight[0, index]*x + decay_value*self.cache + self.bias[index]
        self.cache = out - self.bias[index]
        return out

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Row Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"

class HeadedRepeatCausalLinear(nn.Module):
    """
    Mixed-headed repeat module for ParallelRepeatHeads
    """

    def __init__(self, dim: int, heads: int, head_dim=256, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = Tensor.ones([heads, dim])
        self.bias = Tensor.zeros([heads, dim])
        self.heads = heads
        self.decay_value = Tensor.ones([2, 1]) # N.B. only one value used, for back compatability
        self.decay_constant = decay_constant
        self.cache = Tensor.zeros(heads, head_dim)# first half of cache vectors are row repeat, second half are col repeat

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        x = x.reshape(x.shape[0]//self.heads, x.shape[1], self.n_heads) # Maybe TODO: enforce ints on shapes for x = rearrange(x, '(b h) e -> b e h', h=self.heads)
        decay_value = (decay_value.clip(min=0.9, max=1)**(1/self.decay_constant))
        self.cache = self.cache.permute(1, 0)
        
        # row computation and cache update
        row_out = self.weight[self.heads//2:, index]*x[..., self.heads//2:] + decay_value[1]*self.cache[:, self.heads//2:]
        self.cache[:, self.heads//2:] = row_out

        # col computation and cache update
        col_out = self.weight[:self.heads//2, index]*x[...,:self.heads//2] + self.weight[:self.heads//2, index]*decay_value[1]*self.cache[:, :self.heads//2]
        self.cache[:, :self.heads//2] = col_out / self.weight[:self.heads//2, index]
        
        self.cache = self.cache.permute(1, 0)
        output = F.concat([col_out, row_out], axis=-1)
        output += self.bias[:, index]
        output = output.reshape(x.shape[0]*self.heads, x.shape[1])
        return output

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Multi-Headed (h={self.heads}) Parallel Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"

class ParallelRepeatHeads(nn.Module):

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
        # note that the hidden dim is by definition dim // n_heads
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = max.nn.Linear(dim, dim)
        self.out_proj = max.nn.Linear(dim, dim)
        self.mixer_heads = HeadedRepeatCausalLinear(seq_len, n_heads, decay=decay, decay_constant=seq_len//512)
        self.use_projections = use_projections
        self.head_dim = head_dim
    
    def forward(self, x:torch.Tensor, index: int) -> torch.Tensor:
        batch_dim = x.shape[0]
        if self.use_projections:
            x = self.in_proj(x)
        projections = x.reshape(batch_dim * self.n_heads, self.head_dim) # rearrange(x, "b (h e) -> (b h) e", h=self.n_heads)
        conv_projection = self.mixer_heads(projections, index, head_dim=hidden_dim)
        output = conv_projection.reshape(batch_dim, n_heads * self.head_dim) # rearrange(conv_projection, "(b h) e -> b (h e)", h=self.n_heads)
        if self.use_projections:
            output = self.out_proj(output)
        return output

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Multi-Headed (h={self.heads}) Parallel Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"


class MixedRepeatHeads(nn.Module):

    def __init__(self, dim: int, seq_len: int, hidden_dim: int, n_heads: int, expanded_convs=False, decay=False, use_projections=True):
        super().__init__()
        self.n_heads = n_heads
        self.use_projections = use_projections
        if use_projections:
            self.proj_head = [max.nn.Linear(dim, hidden_dim) for i in range(n_heads)]
            self.out_proj = max.nn.Linear(dim, dim)

        self.hidden_dim = hidden_dim
        self.mixer_heads = [ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)] \
                         + [RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)]

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        activations = []
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim]
            conv_projection = self.mixer_heads[head](projection, index)
            activations.append(conv_projection)

        # concatenate and project multi-headed output
        hidden_layer = F.concat(activations, axis=1) # [b e]
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)

        return hidden_layer

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Multi-Headed (h={self.heads}) Parallel Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"


class RepeatHeads(nn.Module):

    def __init__(self, dim, seq_len, hidden_dim, n_heads, expanded_convs=False, combined_heads=False, use_projections=True, decay=False):
        super().__init__()
        self.n_heads = n_heads
        self.use_projections = use_projections
        self.hidden_dim = hidden_dim
        if self.use_projections:
            self.proj_head = [max.nn.Linear(dim, hidden_dim) for i in range(n_heads)]
            self.out_proj = max.nn.Linear(dim, dim)

        if combined_heads:
            self.mixer_heads = [CombinedRepeatCausalLinear(seq_len, decay=decay, decay_constant=seq_len//512) for i in range(n_heads)]
        else:
            self.mixer_heads = [ColRepeatCausalLinear(seq_len) for i in range(n_heads)]

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        activations = []
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim]

            conv_projection = self.mixer_heads[head](projection, index)
            activations.append(conv_projection)

        # concatenate and project multi-headed output
        hidden_layer = F.concat(activations, axis=1)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)
        return hidden_layer

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Multi-Headed (h={self.heads}) Parallel Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"

class LayerNorm(nn.Module):

    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class MixerBlock(nn.Module):

    def __init__(self, hidden_dim: int, seq_len: int, expansion_factor=4, heads=None, kernel=1, expanded_convs=False, mixed_heads=False, combined_heads=False, decay=False, parallel_heads=False, use_projections=True):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        # TODO: check norm, might have to implement if off basis
        self.channel_norm = LayerNorm(hidden_dim)
        self.token_norm = LayerNorm(hidden_dim)

        # channel-mixing layer
        self.channel_in = max.nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.channel_out = max.nn.Linear(hidden_dim * expansion_factor, hidden_dim)

        if heads is not None and heads > 0:
            if parallel_heads:
                # flat mixer layer
                self.token_mixing_layer = ParallelRepeatHeads(
                    hidden_dim,
                    seq_len, 
                    hidden_dim // heads,
                    heads,
                    use_projections=use_projections, 
                    decay=decay
                ) 
            elif mixed_heads:
                self.token_mixing_layer = MixedRepeatHeads(
                    hidden_dim,
                    seq_len,
                    hidden_dim // heads,
                    heads,
                    use_projections=use_projections,
                    expanded_convs=expanded_convs,
                    decay=decay    
                )
            else:
                self.token_mixing_layer = RepeatHeads(
                    hidden_dim,
                    seq_len,
                    hidden_dim // heads,
                    heads,
                    expanded_convs=expanded_convs,
                    combined_heads=combined_heads,
                    decay=decay
                )  

        else:
            if kernel is not None and kernel > 1:
                self.token_mixing_layer = KernelRepeatLinear(seq_len, kernel=kernel, decay=decay, decay_constant=seq_len//256)
            else:
                self.token_mixing_layer = RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim) 

    def forward(self, x: Tensor, index: int) -> torch.Tensor:
        res = x
        x = self.channel_norm(x)
        x = self.channel_in(x)
        x = F.silu(x)
        x = self.channel_out(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = self.token_mixing_layer(x, index)
        x = x + res
        return x

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Multi-Headed (h={self.heads}) Parallel Repeat Causal Linear Layer, mixing up to {dim} tokens with {embedding_dim} hidden dimension with decay={decay}"


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
        self.input_layer = max.nn.Embedding(vocab_size, dim=hidden_dim)

        self.mixer_blocks = [MixerBlock(
            hidden_dim, seq_len, heads=heads, mixed_heads=mixed_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections
        ) for _ in range(num_blocks)]
        self.output_layer = max.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, index: int, **kwargs):
        print ('model forward pass started')
        input_ids = input_ids[:, -1]
        x = self.input_layer(input_ids)
        # index = index[0]
        for block in self.mixer_blocks:
            x = block(x, index)
        
        logits = self.output_layer(x)
        print ('model forward pass ended')
        return logits

    def __call__(self, x: TensorValue, index: int) -> TensorValue:
        return self.forward(x, index)

    def __repr__(self) -> str:
        return f"Structured Recurrent Mixer Model"

device = driver.CPU()

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab =  len(tokenizer)

    input_string = 'Four score and seven years ago, our'
    input_tokens = tokenizer(input_string, return_tensors='pt').input_ids[:, 1:] # no BOS token
    input_tokens = input_tokens.repeat(2, 1)
    length = input_tokens.shape[1]
    print (input_tokens, length)

    tokenized_length = 512
    dim = 64
    layers = 2
    n_heads = 4
    kernel= 1

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

    # weight_path = f"{checkpoint_root}/..."
    # trained_weights = safe_open(weight_path)
    # model.load_state_dict(trained_weights)
    token_type = TensorType(
        DType.int64, shape=[input_tokens.shape[0], tokenized_length], device=DeviceRef.from_device(device)
    )

    length_type = TensorType(
        DType.int64, shape=[1], device=DeviceRef.from_device(device)
    )
    print (length_type, token_type)

    input_tensor = Tensor.constant(input_tokens, dtype=DType.int64, device=device)
    length = Tensor.constant(length, dtype=DType.int64, device=device)
    compiled_model = model.compile(token_type, length_type)

    start = time.time()
    print ('Model compilation completed')
    output = model(input_tensor, length).to(device)
    print (f'Model forward pass completed in {time.time() - start} seconds')
    print (output.shape)
   
