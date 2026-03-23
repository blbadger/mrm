import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig
import datasets
from datasets import load_from_disk
import os
from dotenv import load_dotenv
import shutil
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig

class ColRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1)) # TODO: revert to ones only
        else:
            self.decay_value = torch.ones(1)
        self.decay_constant = decay_constant
        self.cache = torch.zeros(embedding_dim).to('cuda')

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  b  c  d ]
        [ 0  b  c  d ]
        [ 0  0  c  d ]
        [ 0  0  0  d ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        if self.decay_value is not None:
            M = torch.where(
                j >= i, v[j]*(torch.clip(self.decay_value, min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        else:
            M = torch.where(
                j >= i, v[j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        return M

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out

    def forward(self, x: torch.Tensor, index: int, recurrent: bool) -> torch.Tensor:
        if not recurrent:
            return self._parallel_forward(x)
        self.cache = self.cache.to(x.device)
        decay_value = (torch.clip(self.decay_value, min=0.9, max=1)**(1/self.decay_constant)).to(x.device)
        out = self.weight[0, index]*x + self.weight[0, index]*decay_value*self.cache + self.bias[index]
        self.cache = (out - self.bias[index]) / self.weight[0, index] # cache update: factor out weight, remove bias
        return out

class RowRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=256, decay=False, decay_constant=1, **args):
        super().__init__()
        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
        else:
            self.decay_value = torch.ones(1)
        self.decay_constant = decay_constant
        self.cache = torch.zeros(embedding_dim).to('cuda')

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  a  a  a ]
        [ 0  b  b  b ]
        [ 0  0  c  d ]
        [ 0  0  0  d ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        if self.decay_value is not None:
            M = torch.where(
                j >= i, v[i]*(torch.clip(self.decay_value, min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        else:
            M = torch.where(
                j >= i, v[i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        return M

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out

    def forward(self, x: torch.Tensor, index: int, recurrent: bool) -> torch.Tensor:
        if not recurrent:
            return self._parallel_forward(x)
        self.cache = self.cache.to(x.device)
        # expects x in shape [B, E]
        decay_value = (torch.clip(self.decay_value, min=0.9, max=1)**(1/self.decay_constant)).to(x.device)
        out = self.weight[0, index]*x + decay_value*self.cache + self.bias[index]
        self.cache = out - self.bias[index]
        return out

class CombinedRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, embedding_dim=512, decay=None, decay_constant=1):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
        else:
            self.decay_value = torch.ones(1)
        self.decay_constant = decay_constant
        self.row_cache = torch.zeros(embedding)
        self.col_cache = torch.zeros(embedding)

    def vector_to_colrepeat(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  b  c  d ]
        [ 0  b  c  d ]
        [ 0  0  c  d ]
        [ 0  0  0  d ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        if self.decay_value is not None:
            # NB: 0.95 min, decay_constant=4 for copy
            M = torch.where(
                j >= i, v[j]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        else:
            M = torch.where(
                j >= i, v[j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        return M

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        Wr = self.vector_to_rowrepeat(self.weight[0]).to(x.dtype)
        Wc = self.vector_to_colrepeat(self.weight[1]).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ Wr + x_reshaped @ Wc  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out

    def forward(self, x: torch.Tensor, index: int, recurrent: bool) -> torch.Tensor:
        if not recurrent:
            return self._prallel_forward(x)

        if x.dim() > 2:
            return self.prefill_forward(x)
        B, E, = x.shape
        decay_value = (torch.clip(self.decay_value, min=0.9, max=1)**(1/self.decay_constant)).to(x.device)
        x = x.reshape(B * E, S)  # (B*E, S)
        index = x.shape[-1]
        # row computation and cache update
        row_out = self.weight[0, index]*x[..., index] + decay_value[0]*self.cache # note decay val
        self.row_cache = out

        # col computation and cache update
        col_out = self.weight[0, index]*x[..., index] + self.weight[0, index]*decay_value[1]*self.cache
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
        else:
            self.decay_value = torch.ones(2, 1)
        self.decay_constant = decay_constant
        self.cache = torch.zeros(kernel, embedding_dim)

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        x = x.to(device)
        p = self.kernel-1 # pad value
        B, E, S = x.shape
        decay_value = (torch.clip(self.decay_value[1], min=0.9, max=1)**(1/self.decay_constant)).to(x.device)
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
    def __init__(self, dim: int, heads: int, head_dim=256, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        self.heads = heads
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = torch.ones(2, 1)
        self.cache = torch.zeros(heads, head_dim).to('cuda') # first half of cache vectors are row repeat, second half are col repeat

    def _parallel_forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        if x.dim() > 2:
            return self.prefill_forward(x)
        x = x.to(device) # x has shape [b * h, e]
        x = rearrange(x, '(b h) e -> b e h', h=self.heads)
        decay_value = (torch.clip(self.decay_value, min=0.9, max=1)**(1/self.decay_constant)).to(x.device)
        self.cache = rearrange(self.cache, 'h d -> d h')
        
        # row computation and cache update
        row_out = self.weight[self.heads//2:, index]*x[..., self.heads//2:] + decay_value[1]*self.cache[:, self.heads//2:]
        self.cache[:, self.heads//2:] = row_out

        # col computation and cache update
        col_out = self.weight[:self.heads//2, index]*x[...,:self.heads//2] + self.weight[:self.heads//2, index]*decay_value[1]*self.cache[:, :self.heads//2]
        self.cache[:, :self.heads//2] = col_out / self.weight[:self.heads//2, index]
        
        self.cache = rearrange(self.cache, 'd h -> h d')
        output = torch.cat((col_out, row_out), dim=-1)
        output += self.bias[:, index]
        output = rearrange(output, 'b e h -> (b h) e', h=self.heads)
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
    
    def forward(self, x:torch.Tensor, index: int, *args) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, index: int, recurrent: bool) -> torch.Tensor:
        if recurrent:
            activations = []
            # pre-concatenated out projection
            for head in range(self.n_heads):
                if self.use_projections:
                    projection = self.proj_head[head](x)
                else:
                    projection = x[..., head*self.hidden_dim: (head+1)*self.hidden_dim] # flexible for 2 or 3 dims
                    if torch.is_autocast_enabled():
                        projection = projection.to(torch.float16)

                conv_projection = self.mixer_heads[head](projection, index, recurrent)
                activations.append(conv_projection)

            # concatenate and project multi-headed output
            hidden_layer = torch.cat(activations, dim=1).to(x.dtype)
            if self.use_projections:
                hidden_layer = self.out_proj(hidden_layer)
            return hidden_layer
        else:
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

                conv_projection = self.mixer_heads[head](projection, index, recurrent)
                rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
                activations.append(rearranged_conv)

            # concatenate and project multi-headed output
            hidden_layer = torch.cat(activations, dim=2)
            if self.use_projections:
                hidden_layer = self.out_proj(hidden_layer)

            hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
            return hidden_layer

class RepeatHeads(nn.Module):
    ### Not Implemented Yet
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

    def forward(self, x: torch.Tensor, index: int, *args) -> torch.Tensor:
        activations = []
        # pre-concatenated out projection
        for head in range(self.n_heads):
            if self.use_projections:
                projection = self.proj_head[head](x)
            else:
                projection = x[..., head*self.hidden_dim: (head+1)*self.hidden_dim] # for two or three dims
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

    def __init__(self, 
        hidden_dim: int, 
        seq_len: int, 
        expansion_factor=4, 
        heads=None, 
        kernel=1, 
        expanded_convs=False, 
        mixed_heads=False, 
        combined_heads=False, 
        decay=False, 
        parallel_heads=False, 
        use_projections=True,
        ):

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

    def forward(self, x: torch.Tensor, index: int, recurrent: bool) -> torch.Tensor:
        res = x
        x = self.channel_norm(x)
        x = self.channel_mixing_layer(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = self.token_mixing_layer(x, index, recurrent)
        x = x + res
        return x


class DualMLPMixer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        kernel=1,
        expanded_convs=False,
        mixed_heads=False,
        combined_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True,
        **kwargs
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
                    hidden_dim, 
                    seq_len, 
                    heads=heads, 
                    expanded_convs=expanded_convs, 
                    kernel=kernel, 
                    mixed_heads=mixed_heads, 
                    combined_heads=combined_heads, 
                    decay=decay, 
                    parallel_heads=parallel_heads, 
                    use_projections=use_projections
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._init_weights()
        self.loss_fn = nn.CrossEntropyLoss()

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

    def forward(self, input_ids, index=0, labels=None, **kwargs):
        labels = labels[:, 1:].contiguous()
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x, index, recurrent)
        logits = self.output_layer(x)
        logits = logits[:, :-1].contiguous()

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
        
