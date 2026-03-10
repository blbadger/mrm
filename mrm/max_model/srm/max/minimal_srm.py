from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
import gc

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')


class ColRepeatCausalLinear(Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        # Standard weight + bias
        self.weight = Tensor.zeros([1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self.first_index = 0
        self.cache = torch.zeros([embedding_dim]) # TODO: initialize and send to shared mem via custom op

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        self.weight = self.weight.to(x.device)
        self.bias = self.bias.to(x.device);
        self.decay_value = self.decay_value.to(x.device)
        decay_value = (self.decay_value.clip(min=0.9, max=1)**(1/self.decay_constant))
        out = self.weight[0, index]*x + self.weight[0, index]*decay_value*self.cache + self.bias[index]
        self.cache = (out - self.bias[index]) / self.weight[:, index] # cache update: factor out weight, remove bias
        return out

class RowRepeatCausalLinear(Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        # Standard weight + bias
        self.weight = Tensor.ones([1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
        self.decay_constant = decay_constant
        self.cache = torch.zeros([embedding_dim]) # TODO: initialize and send to shared mem via custom op

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        # expects x in shape [B, E]
        self.weight = self.weight.to(x.device)
        self.bias = self.bias.to(x.device);
        self.decay_value = self.decay_value.to(x.device) 
        decay_value = (self.decay_value.clip(min=0.9, max=1)**(1/self.decay_constant))
        out = self.weight[0, index]*x + decay_value*self.cache + self.bias[index]
        self.cache = out - self.bias[index]
        return out

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
        # note that the hidden dim is by definition dim // n_heads
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

class MixedRepeatHeads(Module):

    def __init__(self, dim: int, seq_len: int, hidden_dim: int, n_heads: int, expanded_convs=False, decay=False, use_projections=True):
        self.n_heads = n_heads
        self.use_projections = use_projections
        if use_projections:
            self.proj_head = max.nn.sequential.ModuleList(max.nn.Linear(dim, hidden_dim) for i in range(n_heads))
            self.out_proj = max.nn.Linear(dim, dim)

        self.hidden_dim = hidden_dim # TODO: replace mixer heads list with module list or sequential, as this is not assigned to device properly
        self.mixer_heads = max.nn.sequential.ModuleList(ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)) \
                         + max.nn.sequential.ModuleList(RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2))
         
        print (self.mixer_heads)

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


class LayerNorm(Module):

    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        normed_out = F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)
        return normed_out


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
        print ('here')
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
                # flatd mixer layer
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
                    decay=decay    
                )
            else:
                self.token_mixing_layer = RepeatHeads(
                    hidden_dim,
                    seq_len,
                    hidden_dim // heads,
                    heads,
                    decay=decay
                )  
        else:
            self.token_mixing_layer = ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim) 

    def forward(self, x) -> torch.Tensor:
        x, index = x
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
        return x, index

class RecurrentSRM(Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        mixed_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.input_layer = max.nn.Embedding(vocab_size, dim=hidden_dim)
        self.mixer_blocks = nn.sequential.ModuleList(MixerBlock(
                hidden_dim, seq_len, heads=heads, mixed_heads=mixed_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections
        ) for _ in range(num_blocks))

    def forward(self, input_ids, index: int):
        print ('model forward pass started')
        index = int(input_ids.shape[-1])
        input_ids = input_ids[:, -1]
        x = self.input_layer(input_ids)
        #for i, block in enumerate(self.mixer_blocks):
        #    x, _ = block((x, index))
        x, _ = self.mixer_blocks((x, index))
        print ('model forward pass ended')
        return x

class SRMLMHeadModel(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        mixed_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True
    ):
        self.model = RecurrentSRM(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, mixed_heads=mixed_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections)
        self.lm_head = Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, index: int):
        model_output = self.model(input_ids, index)
        logits = self.lm_head(model_output)
        return logits

class TotalModel(Module):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def forward(self, input_ids, index: int):
        x = self.model1.model(input_ids, index)
        index = int(input_ids.shape[-1])
        for i, block in enumerate(self.model2.model.mixer_blocks):
            x, _ = block(x, index)
        logits = self.model2.lm_head(x)
        return logits

device = driver.CPU()

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab =  len(tokenizer)

    dtype, device = defaults()
    input_string = 'Four score and seven years ago, our'
    batch_size = 1000
    input_tokens = tokenizer(input_string, return_tensors='pt').input_ids[:, 1].unsqueeze(1) # no BOS token
    input_tokens = input_tokens.repeat(batch_size, 1)
    length = torch.tensor([input_tokens.shape[1]])

    tokenized_length = 512
    dim = 64
    layers = 16
    n_heads = 4
    print (dtype)

    with default_device(CPU()), default_dtype(dtype):
        model = SRMLMHeadModel(
            n_vocab, 
            dim, 
            tokenized_length, 
            layers, 
            heads=n_heads, 
            mixed_heads=True, 
            decay=True, 
            parallel_heads=True, 
            use_projections=False
        )
    
    model = model.to(device)
    # weight_path = f"{checkpoint_root}/..."
    # trained_weights = safe_open(weight_path)
    # model.load_state_dict(trained_weights)

    token_type = TensorType(
        DType.int64, shape=[input_tokens.shape[0], 1], device=DeviceRef.from_device(device)
    )
    length_type = TensorType(
        DType.int64, shape=[1], device=DeviceRef.from_device(device)
    )

    model = model.compile(token_type, length_type)
    print ('compiled')

    with default_device(CPU()), default_dtype(dtype):
        model2 = SRMLMHeadModel(
            n_vocab, 
            dim, 
            tokenized_length, 
            layers, 
            heads=n_heads, 
            mixed_heads=True, 
            decay=True, 
            parallel_heads=False, 
            use_projections=True
        )

    print ('compiled')


    input_tensor = Tensor.constant(input_tokens, dtype=DType.int64, device=device)
    length = Tensor.constant(length, dtype=DType.int64, device=device)

    start = time.time()
    print ('Model compilation completed')
    for i in range(50):
        output = model(input_tensor, length)
    end = time.time()
    print (f'Model throughput: {batch_size / (end - start)} t/s')
    print (output.shape)
   
