import torch
import torch.nn.functional as F
from twa.config import Config
import math
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Removed unused RoPE1D, RoPE3D and RoPE2D classes - only keeping Sinusoidal2DPositionEmbed


class Sinusoidal2DPositionEmbed(nn.Module):
    def __init__(self, dim: int, H: int =Config.blocks_per_col, W: int = Config.blocks_per_row, batch_size:int= Config.batch_size):
        """
        2D sine/cosine positional embedding for H x W grid, flattened to [H*W, D].
        Args:
            dim: embedding dimension, must be divisible by 4
            H, W: spatial resolution
        """
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"

        self.H = H
        self.W = W
        self.dim = dim
        self.batch_size = batch_size

        # === Register CPU-side embedding, move in forward ===
        self.pos_embed = self.build_embedding()

    def build_embedding(self) -> torch.Tensor:
        """
        Compute sin/cos positional embedding of shape [H*W, D] (on CPU).
        """
        grid_y = torch.arange(self.H, device=device) / self.H
        grid_x = torch.arange(self.W, device=device) / self.W
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        pos = torch.stack([mesh_y, mesh_x], dim=-1).reshape(-1, 2)  # [H*W, 2]

        # split embedding dim
        dim_half = self.dim // 2
        emb_y = self._encode(pos[:, 0], dim_half)
        emb_x = self._encode(pos[:, 1], dim_half)

        return torch.cat([emb_y, emb_x], dim=-1)  # [H*W, D]

    def _encode(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Encode 1D normalized positions x ∈ [0, 1] to [len(x), dim] using sine/cosine.
        """
        half_dim = dim // 2
        omega = torch.exp(torch.arange(0, half_dim, device=device) * -(math.log(10000.0) / half_dim))
        angle = x[:, None] * omega[None, :] * math.pi * 2
        return torch.cat([angle.sin(), angle.cos()], dim=-1)  # [L, dim]

    def forward(self, dtype=torch.float16) -> torch.Tensor:
        """
        Returns: [B, H*W, D]
        """
        return self.pos_embed.expand(self.batch_size, -1, -1).to(dtype=dtype, device=device).contiguous()
