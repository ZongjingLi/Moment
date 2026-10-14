# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2026-10-14 15:13:33
# @Last Modified by:   zongjingli
# @Last Modified time: 2026-10-14 15:14:27
import torch
import torch.nn as nn
from model.patchify import PatchGenerator, MaskGenerator
from model.attention import TransformerBlock, MultiHeadAttention

class MAE(nn.Module):
    def __init__(
        self,resolution = (64,64), channels = 3, 
        patch_size = 3, latent_dim = 128, vis_ratio = 0.1, num_heads = 8, pos_dim = 32,
        encoder_layers=4,
        decoder_layers=2,
    ):
        super().__init__()
        W, H = resolution
        K = patch_size
        self.W = W
        self.H = H

        """Compatiblity of Resolution and Patch Size"""
        assert W % K == 0 and H % K == 0
        nW, nH = W // K, H // K
        self.resolution = resolution
        self.channels   = channels
        self.patch_size = patch_size
        self.mask_resolution = (nW, nH)
        self.num_patches = nW * nH
        self.patch_dim = K * K * channels


        """Mask Generator"""
        self.mask_generator = MaskGenerator(resolution=resolution, p = vis_ratio, patch_size=K)

        """Patchify Image Encoder"""
        self.patch_generator = PatchGenerator(patch_size = K)
        self.patch_embed = nn.Linear(self.patch_dim, latent_dim)

        ### mask with poistional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, nW * nH, latent_dim) * 0.02
        )

        # Distinguish previous-frame tokens from target-frame tokens.
        self.frame_embed = nn.Parameter(
            torch.randn(1, 2, 1, latent_dim) * 0.02
        )

        # Replaces the original pixel information of masked target patches.
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, latent_dim)
        )
        nn.init.normal_(self.mask_token, std=0.02)

        """Encode Decode and Reconstruct"""
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=4 * latent_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=encoder_layers
        )

        # Transformer decoder. This is a stack of standard Transformer
        # blocks, rather than the cross-attention decoder used in NLP.
        dec_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=4 * latent_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=decoder_layers
        )

        self.decoder_norm = nn.LayerNorm(latent_dim)

        """Project the Decoded Latent to Patches"""
        self.proj = nn.Linear(latent_dim, patch_size * patch_size * channels)

    def forward(self, x_0, x_t, mask = None):
        """
        Args:
            x_0 : shape of [B,C,W,H]
            x_t : shape of [B,C,W,H]
            mask : optional
        Returns:
            reconstructed x_t and mask
        """
        device = x_0.device
        assert x_0.shape == x_t.shape
        B, C, W, H = x_0.shape

        assert C == self.channels
        assert W == self.W and H == self.H

        x0_patches, (nh, nw) = self.patch_generator(x_0) # B, N, P
        xt_patches, (nh, nw) = self.patch_generator(x_t) # B, N, P
        # generate mask to reduce the infos from xt_patches

        if mask is None: mask = self.mask_generator(B, device).bool() # B, N, 1
        mask = mask.bool()

        mask = (~mask).to(device=x_0.device)
        mask = mask.reshape(B, -1).bool()  # True = masked

        x0 = self.patch_embed(x0_patches)
        xt = self.patch_embed(xt_patches)

        pos = self.pos_embed
        x0 = x0 + pos + self.frame_embed[:, 0]
        xt = xt + pos + self.frame_embed[:, 1]

        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
        xt = torch.where(mask.unsqueeze(-1), mask_tokens + pos
                         + self.frame_embed[:, 1], xt)

        ### Encode X0 Xt with mask ###

        tokens = torch.cat([x0, xt], dim=1)

        encoded = self.encoder(tokens)


        target_encoded = encoded[:, self.num_patches:, :]

        decoded = self.decoder(target_encoded)
        decoded = self.decoder_norm(decoded)



        recon_patches = torch.sigmoid(self.proj(decoded))


        recon_img = self.patch_generator.unpatchify(
            recon_patches, grid = (nh, nw), channels = C)

    
        return recon_img, mask
