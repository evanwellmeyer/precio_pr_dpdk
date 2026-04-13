"""
Small Vision Transformer for pixel-level probabilistic regression.

Drop-in replacement for ProbUNet:
    from vit import ProbViT as ProbUNet
    model = ProbUNet(input_channels, base_channels, kernel_size, p_drop, num_bins, gn_groups=...)

Produces per-pixel softmax probabilities over bins, same as the UNet head.
Designed to be comparable in size to the 64-channel flat UNet (~800K params).
Default config (embed=96 via base_channels, depth=8, heads=6, mlp_ratio=3) ~= 871K params.

Architecture:
    1. Patch embedding (patch_size x patch_size) -> embed_dim tokens
    2. Learnable positional encoding
    3. N transformer encoder blocks (pre-norm, multi-head self-attention)
    4. Reshape tokens back to spatial grid
    5. Light conv decoder to upsample back to full resolution
    6. 1x1 softmax head over bins
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//p, W//p)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, p_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=p_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ProbViT(nn.Module):
    """
    Vision Transformer for dense (pixel-level) probabilistic prediction.

    Same call signature as ProbUNet so it patches into train_pr_dpdk.py:
        model = ProbViT(input_channels, base_channels, kernel_size, p_drop, num_bins, gn_groups=1)
        probs = model(x)          # (B, num_bins, H, W)

    base_channels is repurposed as the transformer embed_dim.
    kernel_size and gn_groups are accepted but unused (for API compat).
    """

    def __init__(
        self,
        input_channels,
        base_channels,
        kernel_size=3,
        p_drop=0.0,
        num_bins=64,
        gn_groups=1,
        pyramid=False,
        # ViT-specific
        patch_size=4,
        depth=8,
        num_heads=6,
        mlp_ratio=3.0,
    ):
        super().__init__()
        embed_dim = base_channels
        self.patch_size = patch_size
        self.num_bins = num_bins

        # --- Patch embedding ---
        self.patch_embed = PatchEmbed(input_channels, embed_dim, patch_size)

        # --- Positional encoding (learnable, for 128x128 / patch_size grid) ---
        # We'll interpolate at runtime if input size differs
        grid_h = 128 // patch_size
        grid_w = 128 // patch_size
        self.num_patches_default = grid_h * grid_w
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches_default, embed_dim) * 0.02)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, p_drop=p_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Upsample decoder (patch tokens -> full resolution) ---
        # Uses a sequence of ConvTranspose2d to go from (H/p, W/p) -> (H, W)
        num_upsample = int(math.log2(patch_size))
        decoder_layers = []
        ch = embed_dim
        for i in range(num_upsample):
            out_ch = ch // 2 if i < num_upsample - 1 else ch // 2
            decoder_layers.extend([
                nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
            ])
            ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)

        # --- Softmax bin head ---
        self.head = nn.Conv2d(ch, num_bins, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def _interpolate_pos(self, num_patches_h, num_patches_w):
        """Interpolate positional embedding if input grid differs from default."""
        N = self.pos_embed.shape[1]
        default_side = int(math.sqrt(N))
        if num_patches_h == default_side and num_patches_w == default_side:
            return self.pos_embed

        pos = self.pos_embed.reshape(1, default_side, default_side, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(num_patches_h, num_patches_w), mode="bilinear", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, num_patches_h * num_patches_w, -1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad to multiple of patch_size
        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")

        Hp, Wp = x.shape[2], x.shape[3]
        gh, gw = Hp // p, Wp // p

        # Patch embed -> (B, embed_dim, gh, gw)
        tokens = self.patch_embed(x)
        # Reshape to sequence: (B, gh*gw, embed_dim)
        tokens = tokens.flatten(2).transpose(1, 2)

        # Add positional encoding
        tokens = tokens + self._interpolate_pos(gh, gw)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Reshape back to spatial: (B, embed_dim, gh, gw)
        tokens = tokens.transpose(1, 2).reshape(B, -1, gh, gw)

        # Upsample to full resolution
        feat = self.decoder(tokens)

        # Softmax head
        logits = self.head(feat)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, pad_h // 2 : H + pad_h // 2, pad_w // 2 : W + pad_w // 2]

        return torch.softmax(logits, dim=1)

    def forward_components(self, x):
        return self.forward(x)
