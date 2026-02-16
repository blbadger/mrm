import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_from_disk
import mlflow
from prettytable import PrettyTable
import os
from dotenv import load_dotenv
import shutil

torch.compiler.reset()

class RepeatCausalLinear(nn.Module):


    def __init__(self, dim: int, heads: int):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        self.heads = heads

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  b  c  d]
        [ 0  b  c  d]
        [ 0  0  c  d]
        [ 0  0  0  d]
        """
        # Expects v is a preformed tensor with shape [k, D]
        m = v.shape[-1] # vector shape

        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        # j - i gives the offset into v. When j < i, we want a 0.
        M = torch.where(
            j >= i, v[..., j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        W = self.vector_to_matrix(self.weight).repeat(x.shape[0]//self.heads, 1, 1) 
        output = torch.bmm(x, W)
        repeated_bias = self.bias.repeat(x.shape[0]//self.heads, 1)
        repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1)
        output += repeated_bias
        return output

class DiagonalColCausalLinear(nn.Module):

    def __init__(self, dim: int, decay=False):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.diag_weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
        else:
            self.decay_value = None

    def vector_to_matrix(self, v: torch.Tensor, v2) -> torch.Tensor:
        """
        [ x  b  c  d ]
        [ 0  y  c  d ]
        [ 0  0  z  d ]
        [ 0  0  0  t ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        v2 = v2.reshape(-1)
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        M = torch.where(
            j > i, v[i]*(torch.clip(self.decay_value[1], min=0.9, max=1)**(j-i)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        M2 = torch.where(
            j == i, v2[j], torch.zeros(m, m, device=v2.device, dtype=v2.dtype)
        )
        M += M2
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, embed_dim, seq_len)
        """
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight, self.diag_weight)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out


class ConstantDiagonalColCausalLinear(nn.Module):

    def __init__(self, dim: int, decay=False):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.diag_weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
        else:
            self.decay_value = None

    def vector_to_matrix(self, v: torch.Tensor, v2) -> torch.Tensor:
        """
        [ x  b  c  d ]
        [ 0  y  c  d ]
        [ 0  0  z  d ]
        [ 0  0  0  t ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        v2 = v2.reshape(-1)
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        M = torch.where(
            j > i, v[i]*(torch.clip(self.decay_value, min=0.9, max=1)**(j-i)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        M2 = torch.where(
            j == i, v2[0], torch.zeros(m, m, device=v2.device, dtype=v2.dtype)
        )
        M += M2
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, embed_dim, seq_len)
        """
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight, self.diag_weight)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out



class ColRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out


class RowRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out

class CombinedRepeatCausalLinear(nn.Module):

    def __init__(self, dim: int, decay=None, decay_constant=1):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

    def vector_to_rowrepeat(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  a  a  a ]
        [ 0  b  b  b ]
        [ 0  0  c  c ]
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
                j >= i, v[i]*(torch.clip(self.decay_value[0], min=0.9, max=1)**((j-i))/self.decay_constant), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        else:
            M = torch.where(
                j >= i, v[i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        return M

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, S = x.shape
        Wr = self.vector_to_rowrepeat(self.weight[0]).to(x.dtype)
        Wc = self.vector_to_colrepeat(self.weight[1]).to(x.dtype)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ Wr + x_reshaped @ Wc  # (B*E, S)
        out = out + self.bias.to(x.dtype)  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out


class KernelRepeatLinear(nn.Module):
    """
    A linear layer with a triangular (causal) mask applied to the weight matrix.
    This ensures each position i cannot use info from positions > i.
    """
    def __init__(self, dim: int, kernel: int, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(kernel, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.kernel = kernel
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:

	   # Expects v is a preformed tensor with shape [k, D]
        m = v.shape[-1] # vector shape

        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        if self.decay_value is not None:
            # NB: 0.95 min, decay_constant=4 for copy
            M = torch.where(
                j >= i, v[..., j]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        else:
            M = torch.where(
                j >= i, v[..., j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        p = self.kernel-1 # pad value
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight)
        # apply pad for k>1 convolution
        padded_x = torch.nn.functional.pad(input=x, pad=(0, 0, p, p), mode='constant', value=0)
        padded_e = padded_x.shape[1]
        processed_x = torch.stack([padded_x[:, i:E + i] for i in range(self.kernel)], dim=1)
        out = processed_x @ W
        accumulated_output = torch.sum(out, dim=1) + self.bias
        return accumulated_output

class HeadedRepeatCausalLinear(nn.Module):
    """
    Mixed-headed repeat module for ParallelRepeatHeads
    """

    def __init__(self, dim: int, heads: int, decay=False, decay_constant=1):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        self.heads = heads
        if decay:
            self.decay_value = nn.Parameter(torch.ones(2, 1))
            self.decay_constant = decay_constant
        else:
            self.decay_value = None

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Given a matrix v of shape (k, m) and head number h >= 0, 
        returns an (k x m x m) matrix M where M[i, j] = v[i] if 
        j >= i, and 0 otherwise.

        For example, if v = [[a, b, c, d], [e, f, g, h]], k=2 then M will be:

        [[
            [ a  a  a  a ]
            [ 0  b  b  b ]
            [ 0  0  c  c ]
            [ 0  0  0  d ]
        ],
        [
            [ a  b  c  d ]
            [ 0  b  c  d ]
            [ 0  0  c  d ]
            [ 0  0  0  d ]
        ]]

        """
        # Expects v is a preformed tensor with shape [k, D]
        m = v.shape[-1] # vector shape

        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        if self.decay_value is not None:
            M = torch.where(
                j >= i, v[:self.heads//2, j]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
            M = torch.cat((M, torch.where(
                j >= i, v[self.heads//2:, i]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )), dim=0)
        else:
            M = torch.where(
                j >= i, v[:self.heads//2, j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )
            M = torch.cat((M, torch.where(
                j >= i, v[self.heads//2:, i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
            )), dim=0)

        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device) # x has shape [b * h, e, t]
        # W has shape [h, t, t] and is repeated to make [b * h, t, t]
        W = self.vector_to_matrix(self.weight).repeat(x.shape[0]//self.heads, 1, 1) 
        output = torch.bmm(x, W)
        repeated_bias = self.bias.repeat(x.shape[0]//self.heads, 1) # [h, t] repeated to [b * h, t]
        repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1) # repeated to [b * h, e, t]
        output += repeated_bias
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
            ).to(device)
            self.out_proj = nn.Linear(dim, dim)

        if expanded_convs:
            self.mixer_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        RepeatCausalLinear(seq_len),
                        nn.SiLU(),
                        RepeatCausalLinear(seq_len),
                    )
                    for i in range(n_heads)
                ]
            )
        else:
            if combined_heads:
                self.mixer_heads = nn.ModuleList(
                    [CombinedRepeatCausalLinear(seq_len, decay=decay, decay_constant=seq_len//512) for i in range(n_heads)]
                ).to(device)
            else:
                self.mixer_heads = nn.ModuleList(
                    [ColRepeatCausalLinear(seq_len) for i in range(n_heads)]
                ).to(device)

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


class MLPMixer(nn.Module):

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

        super(MLPMixer, self).__init__()

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
            or isinstance(m, DiagonalColCausalLinear) or isinstance(m, KernelRepeatLinear) or isinstance(m, HeadedRepeatCausalLinear):
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

def copy_dataset(input_ids):
    n_ctx = len(input_ids[0])
    for i, input in enumerate(input_ids):
        first_half = input[:n_ctx//2]
        copied_halves = torch.cat((first_half, first_half))
        input_ids[i] = copied_halves
    return input_ids

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    tokenized_length = 512
    dim = 1280
    layers = 20
    n_heads = 4
    kernel= 1

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True)
    count_parameters(model)
    #model = torch.compile(model)
    
    n_gpus = torch.cuda.device_count()
    total_batch_size = 64 #  128
    batch_size = total_batch_size // n_gpus
    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"
    output_dir = f"{checkpoint_root}/fineweb_h{n_heads}_decay_nonparallel_mixed_projs_k{kernel}_{dim}_n{layers}_c512_b{batch_size}x{n_gpus}"
  
    datasets.config.IN_MEMORY_MAX_SIZE = 1e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None)
    print(len(train_dataset), len(test_dataset))
    mlflow.end_run()
    print("training begun")
    print(model)
    print (output_dir)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
	warmup_steps=50,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=5e-4,
        fp16=True,
        eval_strategy="steps",
        output_dir=output_dir,
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=True,
        max_steps=200000,
        torch_compile=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # save driver code snapshot in checkpoint dir 
    code_path = os.path.abspath(__file__) 
    if not os.path.isdir(output_dir): 
        os.mkdir(output_dir) 
    shutil.copy(code_path, output_dir) 

    model.train()
    trainer.train()
