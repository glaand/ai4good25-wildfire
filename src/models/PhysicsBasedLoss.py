import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsBasedLoss(nn.Module):
    """
    Physics-informed loss for wildfire spread prediction.

    Components:
      1. BCE loss (data fidelity)
      2. Spatial continuity penalty: fire should spread from nearby active zones
      3. Directional penalty (optional): fire spread aligns with wind & slope

    Args:
        lambda_spatial (float): weight for spatial continuity term
        lambda_directional (float): weight for directional term
        neighborhood_size (int): radius of spatial neighborhood (in pixels)
    """
    def __init__(
        self,
        lambda_spatial: float = 0.2,
        lambda_directional: float = 0.1,
        neighborhood_size: int = 2
    ):
        super().__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_directional = lambda_directional
        self.neighborhood_size = neighborhood_size
        self.bce = nn.BCEWithLogitsLoss()

        # Precompute kernel for morphological dilation
        kernel = torch.ones(
            (1, 1, 2 * neighborhood_size + 1, 2 * neighborhood_size + 1),
            dtype=torch.float32
        )
        self.register_buffer("kernel", kernel)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, x_prev: torch.Tensor = None,
                wind_dir: torch.Tensor = None, slope: torch.Tensor = None):
        """
        Args:
            y_hat (Tensor): predicted logits, shape (B, H, W) or (B, 1, H, W)
            y (Tensor): ground truth fire mask, shape (B, H, W) or (B, 1, H, W)
            x_prev (Tensor, optional): previous-day fire mask, shape (B, 1, H, W)
            wind_dir (Tensor, optional): wind direction in degrees (B, 1, H, W)
            slope (Tensor, optional): slope (B, 1, H, W)
        """
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)

        # BCE data loss
        bce_loss = self.bce(y_hat, y)

        total_loss = bce_loss

        # Only compute physics terms if previous-day fire map is provided
        if x_prev is not None:
            # Morphological dilation of previous fire area
            prev_fire = (x_prev > 0.5).float()
            dilated = F.conv2d(prev_fire, self.kernel, padding=self.neighborhood_size)
            dilated = torch.clamp(dilated, 0, 1)

            pred_probs = torch.sigmoid(y_hat)
            # Penalize predicted fire outside dilated region
            spatial_penalty = torch.mean(F.relu(pred_probs - dilated))

            total_loss += self.lambda_spatial * spatial_penalty

            # Optional directional term if wind/slope provided
            if wind_dir is not None and slope is not None:
                # Compute directional bias mask
                wind_x = torch.cos(torch.deg2rad(wind_dir))
                wind_y = torch.sin(torch.deg2rad(wind_dir))

                # shift previous fire map along wind direction
                shifted = self.shift_tensor(prev_fire, wind_x, wind_y)
                directional_penalty = torch.mean((pred_probs - shifted).pow(2) * F.relu(slope))
                total_loss += self.lambda_directional * directional_penalty

        return total_loss

    @staticmethod
    def shift_tensor(tensor, dx, dy):
        """
        Approximate shift of tensor by dx, dy (wind vector) using bilinear sampling.
        dx, dy in normalized coordinates [-1, 1]
        """
        B, C, H, W = tensor.shape
        # Normalize dx, dy to roughly one pixel = 2/H or 2/W
        dx_norm = dx / (W / 2)
        dy_norm = dy / (H / 2)

        # Create grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=tensor.device),
            torch.linspace(-1, 1, W, device=tensor.device),
            indexing="ij"
        )
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

        new_x = x_grid - dx_norm.squeeze(1)
        new_y = y_grid - dy_norm.squeeze(1)

        grid = torch.stack((new_x, new_y), dim=-1)
        shifted = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return shifted
