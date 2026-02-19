from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import legacy as nn
from max.tensor import Tensor, TensorType, defaults
import max.functional as F
import torch


class ColRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        # Standard weight + bias
        self.weight = Tensor.zeros( [1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
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
        self.weight = Tensor([1, dim]) # init to randn
        self.bias = Tensor.zeros([dim]) # init to zero
        self.decay_value = Tensor.ones([1])
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
        self.weight = Tensor([heads, dim])
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
        output = F.concat([col_out, row_out], dim=-1)
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
        hidden_dim: int,
        n_heads: int,
        use_projections=True,
        decay=False,
        **kwargs
    ):
        # note that the hidden dim is by definition dim // n_heads
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.mixer_heads = HeadedRepeatCausalLinear(seq_len, n_heads, decay=decay, decay_constant=seq_len//512)
        self.use_projections = use_projections
    
    def forward(self, x:torch.Tensor, index: int) -> torch.Tensor:
        if self.use_projections:
            x = self.in_proj(x)
        projections = rearrange(x, "b (h e) -> (b h) e", h=self.n_heads)
        conv_projection = self.mixer_heads(projections, index)
        output = rearrange(conv_projection, "(b h) e -> b (h e)", h=self.n_heads)
        if self.use_projections:
            output = self.out_proj(output)
        return output

class MixedRepeatHeads(nn.Module):

    def __init__(self, dim: int, seq_len: int, hidden_dim: int, n_heads: int, expanded_convs=False, decay=False, use_projections=True):
        super().__init__()
        self.n_heads = n_heads
        self.use_projections = use_projections
        if use_projections:
            self.proj_head = nn.ModuleList(
                [nn.Linear(dim, hidden_dim) for i in range(n_heads)]
            )
            self.out_proj = nn.Linear(dim, dim)

        self.hidden_dim = hidden_dim
        self.mixer_heads = nn.ModuleList(
            [ColRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)] + [RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)]
        )

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        activations = []
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim]
                if torch.is_autocast_enabled():
                    projection = projection.to(torch.float16)

            conv_projection = self.mixer_heads[head](projection, index)
            activations.append(conv_projection)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=1)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)

        return hidden_layer

class RepeatHeads(nn.Module):

    def __init__(self, dim, seq_len, hidden_dim, n_heads, expanded_convs=False, combined_heads=False, use_projections=True, decay=False):
        super().__init__()
        self.n_heads = n_heads
        self.use_projections = use_projections
        self.hidden_dim = hidden_dim
        if self.use_projections:
            self.proj_head = nn.ModuleList(
                [nn.Linear(dim, hidden_dim) for i in range(n_heads)]
            )
            self.out_proj = nn.Linear(dim, dim)

        if combined_heads:
            self.mixer_heads = nn.ModuleList(
                [CombinedRepeatCausalLinear(seq_len, decay=decay, decay_constant=seq_len//512) for i in range(n_heads)]
            )
        else:
            self.mixer_heads = nn.ModuleList(
                [ColRepeatCausalLinear(seq_len) for i in range(n_heads)]
            )

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        activations = []
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim]
                if torch.is_autocast_enabled():
                    projection = projection.to(torch.float16)

            conv_projection = self.mixer_heads[head](projection, index)
            activations.append(conv_projection)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=1)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)
        return hidden_layer


class MixerBlock(nn.Module):

    def __init__(self, hidden_dim: int, seq_len: int, expansion_factor=4, heads=None, kernel=1, expanded_convs=False, mixed_heads=False, combined_heads=False, decay=False, parallel_heads=False, use_projections=True):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        # channel-norm
        self.channel_norm = nn.LayerNorm(hidden_dim)

        # channel-mixing layer
        self.channel_mixing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

        # token-norm
        self.token_norm = nn.LayerNorm(hidden_dim)
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
            if expanded_convs:
                # token-mixing layer
                self.token_mixing_layer = nn.Sequential(
                    RepeatCausalLinear(seq_len),
                    nn.SiLU(),
                    RepeatCausalLinear(seq_len),
                ) 
            elif kernel is not None and kernel > 1:
                self.token_mixing_layer = KernelRepeatLinear(seq_len, kernel=kernel, decay=decay, decay_constant=seq_len//256)
            else:
                self.token_mixing_layer = RowRepeatCausalLinear(seq_len, embedding_dim=hidden_dim) 

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        res = x
        x = self.channel_norm(x)
        x = self.channel_mixing_layer(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = self.token_mixing_layer(x, index)
        x = x + res
        return x


class RecurrentMLPMixer(nn.Module):

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
        self.input_layer = nn.Embedding(vocab_size, hidden_dim)

        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    hidden_dim, seq_len, heads=heads, mixed_heads=mixed_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

        self.loss_fn = nn.CrossEntropyLoss()
        self.copy = copy


    def forward(self, input_ids, index: int, labels=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            if labels is not None:
                labels = copy_dataset(labels)
        labels = labels[:, 1:].contiguous()
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x, index)
        logits = self.output_layer(x)
        logits = logits[:, :-1].contiguous()

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits


if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    input_string = 'Four score and seven years ago, our'
    input_tokens = tokenizer(input_string, return_tensors='pt')

    tokenized_length = 512
    dim = 1024
    layers = 16
    n_heads = 4
    kernel= 1

    model = MLPMixer(
        n_vocab, 
        dim, 
        tokenized_length, 
        layers, 
        heads=n_heads, 
        copy=False, 
        mixed_heads=True, 
        combined_heads=False,
        decay=True, 
        parallel_heads=False, 
        use_projections=True
    )

    model.load_state_dict(open(f"{checkpoint_root}/fineweb_h4_decay_mixedrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors"))
    input_type = TensorType(DType.int, [input_tokens.shape[0], input_tokens.shap[e[1]]])
    input_tensor = Tensor.constant(input_tokens, dtype=DType.int, device=driver.CPU()).to(driver.Accelerator())
    compiled_model = model.compile(input_type)
    output = compiled_model(input_tokens)
   
