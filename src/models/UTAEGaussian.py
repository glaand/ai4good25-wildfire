from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class SoftBinarizer(nn.Module):
    """Differentiable binarization layer with a learnable threshold and sharpness."""
    def __init__(self, init_thresh=0.5, init_scale=10.0):
        super().__init__()
        self.logit_thresh = nn.Parameter(torch.logit(torch.tensor(init_thresh)))
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        # Sigmoid with learnable threshold
        return torch.sigmoid(self.scale * (x - torch.sigmoid(self.logit_thresh)))


class UTAEGaussian(BaseModel):
    """UTAE variant that outputs a continuous fire likelihood map, with optional differentiable binarization."""
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=True,
            *args,
            **kwargs
        )

        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )
        self.binarizer = SoftBinarizer(init_thresh=0.5, init_scale=10.0)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        out = self.model(x, batch_positions=doys, return_att=False)

        # --- Fix: ensure output tensor shape is [B, 1, H, W] ---
        if isinstance(out, (tuple, list)):
            out = out[0]  # some UTAE versions return (pred, att)
        elif isinstance(out, dict):
            out = out["out"] if "out" in out else list(out.values())[0]

        # If UTAE returns [B, T, H, W] or [B, 1, H, W], fix it
        if out.ndim == 3:
            out = out.unsqueeze(1)  # -> [B, 1, H, W]
        elif out.ndim == 5:
            # sometimes output is [B, T, C, H, W], take last timestep or average
            out = out[:, -1, 0:1, :, :]

        return self.binarizer(out)
