import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from einops import rearrange

from rift_svc.utils import (
    exists, 
    lens_to_mask,
) 


class RF(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        odeint_kwargs: dict = dict(
            method='euler'
        ),
        cvec2_drop_prob: float = 0.2,
        num_mel_channels: int | None = 128,
        lognorm: bool = True,
    ):
        super().__init__()

        self.num_mel_channels = num_mel_channels

        self.cvec2_drop_prob = cvec2_drop_prob

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Sampling related parameters
        self.odeint_kwargs = odeint_kwargs

        self.mel_min = -12
        self.mel_max = 2

        self.lognorm = lognorm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        src_mel: torch.Tensor,           # [b n d]
        spk_id: torch.Tensor,        # [b]
        f0: torch.Tensor,            # [b n]
        rms: torch.Tensor,           # [b n]
        cvec: torch.Tensor,          # [b n d]
        cvec2: torch.Tensor,         # [b n d2]
        frame_len: torch.Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 2.,
        seed: int | None = None,
        interpolate_condition: bool = False,
        t_inter: float = 0.0,
    ):
        self.eval()

        batch, mel_seq_len, device = *src_mel.shape[:2], src_mel.device

        if not exists(frame_len):
            frame_len = torch.full((batch,), mel_seq_len, device=device)

        mask = lens_to_mask(frame_len)

        # Define the ODE function
        def fn(t, x):
            null_pred = self.transformer(
                x=x, 
                spk=spk_id, 
                f0=f0, 
                rms=rms, 
                cvec=cvec, 
                cvec2=cvec2,
                time=t, 
                drop_cvec2=True, 
                mask=mask
            )
            if cfg_strength < 1e-5:
                return null_pred

            pred = self.transformer(
                x=x, 
                spk=spk_id, 
                f0=f0, 
                rms=rms, 
                cvec=cvec, 
                cvec2=cvec2,
                time=t, 
                drop_cvec2=False,
                mask=mask
            )

            #return pred + (pred - null_pred) * cfg_strength
            return null_pred + (pred - null_pred) * cfg_strength

        # Noise input
        y0 = []
        for _ in range(batch):
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(mel_seq_len, self.num_mel_channels, device=self.device))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # Handle duplicate test case
        if interpolate_condition:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * self.norm_mel(src_mel)
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps, device=self.device)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        
        sampled = trajectory[-1]
        out = self.denorm_mel(sampled)
        out = torch.where(mask.unsqueeze(-1), out, src_mel)

        return out, trajectory

    def forward(
        self,
        inp: torch.Tensor,        # mel
        spk_id: torch.Tensor,     # [b]
        f0: torch.Tensor,         # [b n]
        rms: torch.Tensor,        # [b n]
        cvec: torch.Tensor,       # [b n d]
        cvec2: torch.Tensor,      # [b n d2]
        frame_len: torch.Tensor | None = None,
    ):
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # Handle lengths and masks
        if not exists(frame_len):
            frame_len = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(frame_len, length=seq_len)  # Typically padded to max length in batch

        x1 = self.norm_mel(inp)
        x0 = torch.randn_like(x1)

        if self.lognorm:
            quantiles = torch.linspace(0, 1, batch + 1).to(x1.device)
            z = quantiles[:-1] + torch.rand((batch,)).to(x1.device) / batch
            # now transform to normal
            z = torch.erfinv(2 * z - 1) * math.sqrt(2)
            time = torch.sigmoid(z)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)

        t = rearrange(time, 'b -> b 1 1')
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # unconditional guiding dropout rates
        #drop_whisper = torch.rand((batch,), device=device) < self.whisper_drop_prob
        # Drop a fixed proportion of the batch
        num_to_drop = int(batch * self.cvec2_drop_prob)
        drop_indices = torch.randperm(batch, device=device)[:num_to_drop]
        drop_cvec2 = torch.zeros(batch, dtype=torch.bool, device=device)
        drop_cvec2[drop_indices] = True

        pred = self.transformer(
            x=xt, 
            spk=spk_id, 
            f0=f0, 
            rms=rms, 
            cvec=cvec, 
            cvec2=cvec2,
            time=time, 
            drop_cvec2=drop_cvec2,
            mask=mask
        )

        # Flow matching loss
        loss = F.mse_loss(pred, flow, reduction='none')
        loss = loss[mask]

        return loss.mean(), pred

    def norm_mel(self, mel: torch.Tensor):
        return (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1
    
    def denorm_mel(self, mel: torch.Tensor):
        return (mel + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min
