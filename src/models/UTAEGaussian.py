from typing import Any

import torch
import torch.nn as nn

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class SoftBinarizer(nn.Module):
    """Differentiable binarization layer with a learnable threshold and sharpness."""
    def __init__(self, init_thresh=0.5, init_scale=10.0):
        super().__init__()
        self.logit_thresh = nn.Parameter(torch.logit(torch.tensor(init_thresh)))
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        # Sigmoid with learned threshold and scale (sharpness)
        return torch.sigmoid(self.scale * (x - torch.sigmoid(self.logit_thresh)))


class UTAEGaussian(BaseModel):
    """_summary_ U-Net architecture with temporal attention in the bottleneck and skip connections.
    """
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
            use_doy=True, # UTAE uses the day of the year as an input feature
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
        out1 = self.model(x, batch_positions=doys, return_att=False)
        print(f"Shape of output 1: {out1.shape}")
        out2 = self.binarizer(out1)
        print("This is a test")
        print(f"Shape of output 2: {out2.shape}")
        print(out2)
        print("test done")
        return out2
