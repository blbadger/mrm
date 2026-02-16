import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_from_disk
import mlflow
import os
from dotenv import load_dotenv
import shutil


class ColRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, **args):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
        else:
            self.decay_value = torch.ones(1)
        self.cache = torch.zeros(embedding_dim) # put on device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        decay_value = torch.clip(self.decay_value, min=0.9, max=1).to(x.device)
        self.cache = self.cache.to(x.device)
        x = x.reshape(B * E, S)  # (B*E, S)
        index = x.shape[-1] - 1 # TODO: pass index from high level, no way of knowing here
        out = self.weight[0, index]*decay_value*x[..., index] + self.weight[0, index]*decay_value*self.cache + self.bias[index]
        self.cache = (out - self.bias[index]) / self.weight[0, index] # cache update: factor out weight, remove bias
        x[..., -1] = out
        out = x
        out = out.view(B, E, S)
        return out


class RowRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, **args):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
        else:
            self.decay_value = torch.ones(1)
        self.cache = torch.zeros(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        decay_value = torch.clip(self.decay_value, min=0.9, max=1).to(x.device)
        self.cache = self.cache.to(x.device)
        x = x.reshape(B * E, S)  # (B*E, S)
        index = x.shape[-1] - 1
        out = self.weight[0, index]*decay_value*x[..., index] + decay_value*self.cache + self.bias[index]
        self.cache = out - self.bias[index]
        x[..., -1] = out
        out = x
        out = out.view(B, E, S)  # reshape back
        return out

class CombinedRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=512, decay=None, decay_constant=1):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None
        self.row_cache = torch.zeros(embedding)
        self.col_cache = torch.zeros(embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        Wr = self.vector_to_rowrepeat(self.weight[0]).to(x.dtype)
        Wc = self.vector_to_colrepeat(self.weight[1]).to(x.dtype)
        x = x.reshape(B * E, S)  # (B*E, S)
        index = x.shape[-1]
        # row computation and cache update
        row_out = self.decay*self.weight[0, index]*x + self.decay*self.cache
        self.row_cache = row_out

        # col computation and cache update
        col_out = self.weight[1, index]*self.decay_value*x + self.weight[1, index]*self.decay_value*self.cache 
        self.col_cache = out / self.weight[index]

        out = row_out + col_out + self.bias[index]  # (B*E, S)
        out = out.view(B, E, S)  # reshape back
        return out


class KernelRepeatLinear(nn.Module):

    def __init__(self, dim: int, kernel: int, embedding_dim=512, decay=False, decay_constant=1):
        
        # column repeat kernel mixer
        super().__init__()

        self.weight = nn.Parameter(torch.randn(kernel, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.kernel = kernel
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

        self.cache = torch.zeros(kernel, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        p = self.kernel-1 # pad value
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight)
        # apply pad for k>1 convolution
        padded_x = torch.nn.functional.pad(input=x, pad=(0, 0, p, p), mode='constant', value=0)
        padded_e = padded_x.shape[1]
        processed_x = torch.stack([padded_x[:, i:E + i] for i in range(self.kernel)], dim=1)
         # col computation and cache update
        out = self.weight[:, index]*self.decay_value*x + self.weight[:, index]*self.decay_value*self.cache 
        self.cache = out / self.weight[:, index]
        accumulated_output = torch.sum(out, dim=1) + self.bias
        return accumulated_output

class HeadedRepeatCausalLinear(nn.Module):
    """
    Mixed-headed repeat module for ParallelRepeatHeads
    """

    def __init__(self, dim: int, heads: int, head_dim=128, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        self.heads = heads
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = 1
        self.cache = torch.zeros(heads, head_dim) # first half of cache vectors are row repeat, second half are col repeat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device) # x has shape [b * h, e, t]
        # row computation and cache update
        rows_out = self.decay*self.weight[:self.heads//2][index]*x[:self.heads//2] + self.decay*self.cache
        self.cache[:self.heads//2] = rows_out

        # col computation and cache update
        cols_out = self.weight[self.heads//2:][index]*self.decay_value*x + self.weight[self.heads//2:][index]*self.decay_value*self.cache 
        self.cache[self.heads//2:] = out / self.weight[index]
        
        output = torch.cat((rows_out, cols_out), dim=0)
        output += self.bias[:, index]
        return output

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
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b e t -> b t e")
        if self.use_projections:
            x = self.in_proj(x)
        projections = rearrange(x, "b t (h e) -> (b h) e t", h=self.n_heads)
        conv_projection = self.mixer_heads(projections)
        output = rearrange(conv_projection, "(b h) e t -> b t (h e)", h=self.n_heads)
        if self.use_projections:
            output = self.out_proj(output)
        output = rearrange(output, "b t e -> b e t")
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
            [ColRepeatCausalLinear(seq_len, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)] + [RowRepeatCausalLinear(seq_len, decay=decay, decay_constant=seq_len//512) for i in range(n_heads//2)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        if self.use_projections:
            x = rearrange(x, "b e t -> b t e")
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
                projection = rearrange(projection, "b t e -> b e t")
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim, :]
                if torch.is_autocast_enabled():
                    projection = projection.to(torch.float16)

            conv_projection = self.mixer_heads[head](projection)
            rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
            activations.append(rearranged_conv)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=2)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)

        hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        if self.use_projections:
            x = rearrange(x, "b e t -> b t e")
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
                projection = rearrange(projection, "b t e -> b e t")
            else:
                projection = x[:, head*self.hidden_dim: (head+1)*self.hidden_dim, :]
                if torch.is_autocast_enabled():
                    projection = projection.to(torch.float16)

            conv_projection = self.mixer_heads[head](projection)
            rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
            activations.append(rearranged_conv)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=2)
        if self.use_projections:
            hidden_layer = self.out_proj(hidden_layer)
        hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
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
                self.token_mixing_layer = ColRepeatCausalLinear(seq_len) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.channel_norm(x)
        x = self.channel_mixing_layer(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)
        x = self.token_mixing_layer(x)
        x = x.transpose(1, 2)
        x = x + res
        return x


class CachedMLPMixer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        kernel=1,
	    expanded_convs=False,
        copy=False,
        mixed_heads=False,
        combined_heads=False,
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
                    hidden_dim, seq_len, heads=heads, expanded_convs=expanded_convs, kernel=kernel, mixed_heads=mixed_heads, combined_heads=combined_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

        self.loss_fn = nn.CrossEntropyLoss()
        self.copy = copy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, ColRepeatCausalLinear) or isinstance(m, RowRepeatCausalLinear) or isinstance(m, CombinedRepeatCausalLinear) \
            or isinstance(m, KernelRepeatLinear) or isinstance(m, HeadedRepeatCausalLinear):
                # Kaiming He initialization for Swish activation
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            if labels is not None:
                labels = copy_dataset(labels)
        labels = labels[:, 1:].contiguous()
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        logits = logits[:, :-1].contiguous()

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    tokenized_length = 1024
    dim = 1024
    layers = 16
    n_heads = 4
    kernel= 1

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True)

   
