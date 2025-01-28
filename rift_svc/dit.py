import math
from typing import Union

from einops import repeat
from jaxtyping import Bool, Float, Int
import torch
from torch import nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from rift_svc.modules import (
    AdaLayerNormZero_Final,
    DiTBlock,
    MLP,
    TimestepEmbedding,
    #ConvLinear,
)


# Conditional embedding for f0, rms, cvec
class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim: int, rspin_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        #self.rspin_dim = rspin_dim
        self.cond_dim = cond_dim

        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        #self.rspin_embed = nn.Linear(rspin_dim, cond_dim)

        self.mlp = MLP(cond_dim, cond_dim, mult = 2)

        self.ln_cvec = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        #self.ln_rspin = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, elementwise_affine=True, eps=1e-6)

    def forward(
            self,
            f0: Float[torch.Tensor, "b n"],
            rms: Float[torch.Tensor, "b n"],
            cvec: Float[torch.Tensor, "b n d"],
            #rspin: Float[torch.Tensor, "b n d2"]
        ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        # f0_embed = self.f0_embed(f0 / 1200)
        # rms_embed = self.rms_embed(rms)
        f0_embed = self.f0_embed(f0 / 1200)
        rms_embed = self.rms_embed(rms)
        cvec_embed = self.ln_cvec(self.cvec_embed(cvec))
        #rspin_embed = self.ln_rspin(self.rspin_embed(rspin))

        cond = f0_embed + rms_embed + cvec_embed# + rspin_embed
        cond = self.mlp(cond) + cond
        return self.ln(cond)


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.mlp = MLP(out_dim, out_dim, mult = 2)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Float[torch.Tensor, "b n d1"], cond_embed: Float[torch.Tensor, "b n d2"]):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim = -1)
        x = self.proj(x)
        x = self.mlp(x) + x
        x = self.ln(x)
        return x


# backbone using DiT blocks
class DiT(nn.Module):
    def __init__(self,
                 dim: int, depth: int, head_dim: int = 64, dropout: float = 0.1, ff_mult: int = 4,
                 mel_dim: int = 128, num_speaker: int = 1, cvec_dim: int = 768, rspin_dim: int = 768,
                 init_std: float = 1):
        super().__init__()

        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.time_embed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, rspin_dim, dim)
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    head_dim = head_dim,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.output = nn.Linear(dim, mel_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.attn_norm.proj.weight, 0)
            torch.nn.init.constant_(block.attn_norm.proj.bias, 0)

        torch.nn.init.constant_(self.cond_embed.mlp.mlp_out.weight, 0)
        torch.nn.init.constant_(self.input_embed.mlp.mlp_out.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.bias, 0)
        torch.nn.init.constant_(self.output.weight, 0)
        torch.nn.init.constant_(self.output.bias, 0)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            # Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813).
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # weight shape: (out_channels, in_channels/groups, kernel_size)
            fan_out = module.weight.shape[0]  # out_channels
            fan_in = module.weight.shape[1] * module.weight.shape[2]  # (in_channels/groups) * kernel_size
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std/math.sqrt(self.dim))


    def forward(
        self,
        x: Float[torch.Tensor, "b n d1`"],  # nosied input mel
        spk: Int[torch.Tensor, "b"],  # speaker
        f0: Float[torch.Tensor, "b n"],
        rms: Float[torch.Tensor, "b n"],
        cvec: Float[torch.Tensor, "b n d2"],
        #rspin: Float[torch.Tensor, "b n d3"],
        time: Float[torch.Tensor, "b"] | Float[torch.Tensor, "b n"],  # time step
        mask: Bool[torch.Tensor, "b n"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)

        t = self.time_embed(time)
        spk_embeds = self.spk_embed(spk)
        t = t + spk_embeds

        cond_embed = self.cond_embed(f0, rms, cvec)
        x = self.input_embed(x, cond_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask = mask, rope = rope)

        x = self.norm_out(x, t)
        output = self.output(x)

        return output
