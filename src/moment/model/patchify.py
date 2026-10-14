# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2026-10-14 15:13:11
# @Last Modified by:   zongjingli
# @Last Modified time: 2026-10-14 15:13:12
import torch
import torch.nn as nn

class PatchGenerator(nn.Module):
    def __init__(self, patch_size=2):
        super().__init__()
        self.k = patch_size

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        k = self.k

        assert H % k == 0 and W % k == 0
        nh, nw = H // k, W // k

        # [B, C, nh, nw, k, k]
        patches = x.unfold(2, k, k).unfold(3, k, k)
        # [B, nh, nw, C, k, k]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # [B, nh * nw, C * k * k]
        patches = patches.reshape(B, nh * nw, C * k * k)

        return patches, (nh, nw)

    def unpatchify(self, patches, grid, channels):
        # patches: [B, N, C * k * k]
        B, N, _ = patches.shape
        nh, nw = grid
        k = self.k

        assert N == nh * nw, f"N:{N} and {nh}x{nw}"
        x = patches.reshape(B, nh, nw, channels, k, k)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(B, channels, nh * k, nw * k)

class MaskGenerator:
    def __init__(self, resolution=(64, 64), p=0.1, patch_size=2):
        self.resolution = resolution
        self.p = p
        self.patch_size = patch_size

        self.num_patches_h = resolution[0] // patch_size
        self.num_patches_w = resolution[1] // patch_size

    def __call__(self, batch_size, device=None):
        num_patches = self.num_patches_h * self.num_patches_w

        return torch.bernoulli(
            torch.full(
                (batch_size, num_patches, 1),
                self.p,
                device=device,))