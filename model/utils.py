import io
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Int
from PIL import Image


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v: Any) -> bool:
    return v is not None

def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


def draw_mel_specs(gt: np.ndarray, gen: np.ndarray, diff: np.ndarray, cache_path: str):
    vmin = min(gt.min(), gen.min(), diff.min())
    vmax = max(gt.max(), gen.max(), diff.max())
    
    # Create figure with space for colorbar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'hspace': 0})
    
    # Plot all spectrograms with the same scale
    im1 = ax1.imshow(gt, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('GT', fontsize=14)
    ax1.set_xticks([])
    
    im2 = ax2.imshow(gen, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Gen', fontsize=14)
    ax2.set_xticks([])
    
    im3 = ax3.imshow(diff, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_ylabel('Diff', fontsize=14)
    
    # Add single shared colorbar
    fig.colorbar(im1, ax=[ax1, ax2, ax3], location='right')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Open with PIL and save as compressed JPEG
    img = Image.open(buf)
    img = img.convert('RGB')
    img.save(cache_path, 'JPEG', quality=85, optimize=True)
    buf.close()



# tensor helpers

def lens_to_mask(
    t: Int[torch.Tensor, "b"],
    length: int | None = None
) -> Bool[torch.Tensor, "b n"]: 

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return seq < t[..., None]