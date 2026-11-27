from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from slot_attention.slot_attention import SlotAttention
from slot_attention.multi_head_slot_attention import MultiHeadSlotAttention

# functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_softmax(logits, temperature = 1.):
    dtype, size = logits.dtype, logits.shape[-1]

    assert temperature > 0

    scaled_logits = logits / temperature

    # gumbel sampling and derive one hot

    noised_logits = scaled_logits + gumbel_noise(scaled_logits)

    indices = noised_logits.argmax(dim = -1)

    hard_one_hot = F.one_hot(indices, size).type(dtype)

    # get soft for gradients

    soft = scaled_logits.softmax(dim = -1)

    # straight through

    hard_one_hot = hard_one_hot + soft - soft.detach()

    # return indices and one hot

    return hard_one_hot, indices

def get_slotattention_decoder_backbone(object_dim: int, output_dim: int = 4):
    """Slot Attention decoder that upsamples 1 x 1 to 32 x 32 with five stride-2 layers."""
    return nn.Sequential(
        # Input shape [B, C, 1, 1].
        nn.ConvTranspose2d(object_dim, 64, 5, stride=2, padding=2, output_padding=1),  # → 2×2
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),        # → 4×4
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),              # → 8×8
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),        # → 16×16
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),        # → 32×32
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim + 1, 3, stride=1, padding=1),              # → 32×32，+1 for alpha
    )


class PatchDecoderCNN(nn.Module):

    def __init__(
        self,
        object_dim: int,
        output_dim: int,        # e.g. 768
        grid_size=(32, 32),
        decoder_input_dim: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.num_patches = self.H * self.W

        # Slot projection: 768 → 256
        self.inp_transform = nn.Linear(object_dim, decoder_input_dim)
        nn.init.xavier_uniform_(self.inp_transform.weight)
        nn.init.zeros_(self.inp_transform.bias)

        # CNN decoder: 1×1 → 32×32
        self.decoder = get_slotattention_decoder_backbone(
            decoder_input_dim,
            output_dim
        )

    def forward(self, slots, keep_mask=None):
        B, K, _ = slots.shape

        # (B*K, C)
        x = self.inp_transform(slots).flatten(0, 1)

        # → (B*K, C, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # CNN upsampling: output (B*K, D+1, 32, 32)
        x = self.decoder(x)

        # separate feature + alpha
        feat_map, alpha_map = x.split([self.output_dim, 1], dim=1)

        # reshape back to B,K
        feat_map = feat_map.unflatten(0, (B, K))
        alpha_map = alpha_map.unflatten(0, (B, K))

        # Slot existence mask
        if keep_mask is not None:
            keep_mask = keep_mask.view(B, K, 1, 1, 1)
            alpha_map = alpha_map * keep_mask

        # slot softmax
        alpha_map = alpha_map.softmax(dim=1)

        # weighted blending
        recon = (feat_map * alpha_map).sum(dim=1)   # [B, D, 32, 32]

        # flatten to patch sequence
        recon = rearrange(recon, 'b d h w -> b (h w) d')

        return recon
    

class SlotSuperVLAD(nn.Module):
    """
    Map slots selected by cross-view matching to a 3072-dimensional global descriptor.
    """
    def __init__(
        self,
        slot_dim: int = 256,
        num_clusters: int = 4,           # A common setting uses 64 clusters and 48 dimensions, giving 3072 features.
        desc_dim: int = 3072,
        ghost_norm: bool = True,          # SuperVLAD normalization option.
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.ghost_norm = ghost_norm

        # Learnable cluster centers form the NetVLAD codebook.
        self.centroids = nn.Parameter(torch.randn(num_clusters, slot_dim))

        # Project slots into the residual space before aggregation.
        self.project = nn.Linear(slot_dim, slot_dim, bias=False)
        nn.init.orthogonal_(self.project.weight)  # Orthogonal initialization

        # Adjust the output dimension, optionally reducing slot_dim from 256 to 48.
        final_cluster_dim = desc_dim // num_clusters
        self.final_proj = nn.Linear(slot_dim, final_cluster_dim) if final_cluster_dim != slot_dim else nn.Identity()

    def forward(self, slots: torch.Tensor, keep_mask: torch.Tensor):
        """
        slots:      [B, K, D]
        keep_mask: [B, K] float mask, where 1 indicates a cross-view match.
        Returns an L2-normalized descriptor of shape [B, 3072].
        """
        B, _, D = slots.shape

        # 1. Retain slots with successful matches.
        valid_slots = slots[keep_mask > 0.5]          # [N_valid, D]
        if valid_slots.shape[0] == 0:                 # Handle the case with no matched slots.
            return torch.zeros(B, self.num_clusters * (D if hasattr(self.final_proj, 'out_features') else D),
                              device=slots.device)

        # 2. Project into the residual space.
        residuals = self.final_proj(self.project(valid_slots))   # [N_valid, final_dim]

        # 3. Compute soft cluster assignments.
        dist = torch.cdist(residuals, self.centroids)            # [N_valid, C]
        soft_assign = F.softmax(-dist * 10.0, dim=-1)            # τ=0.1

        # 4. Aggregate one residual vector per VLAD cluster.
        vlad = torch.zeros(B, self.num_clusters, residuals.shape[-1], device=slots.device)
        batch_idx = (keep_mask > 0.5).nonzero(as_tuple=True)[0]

        # Accumulate residuals.
        for c in range(self.num_clusters):
            res_c = residuals - self.centroids[c]
            weighted = soft_assign[:, c:c+1] * res_c              # [N_valid, dim]
            vlad[batch_idx] += weighted.unsqueeze(0).sum(dim=1)  # Accumulate within each batch item.

        # 5. Apply SuperVLAD normalization.
        vlad = F.normalize(vlad, p=2, dim=-1)                    # intra-norm
        vlad = vlad.view(B, -1)
        vlad = F.normalize(vlad, p=2, dim=-1)                      # L2 norm

        return vlad  # [B, 3072]

class CVGLSlotVLAD(nn.Module):
    def __init__(
        self,
        slot_attention_g,
        slot_attention_s,
        slot_dim=768,
        vlad_dim=3072,
        temperature=0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.slot_attn_g = slot_attention_g
        self.slot_attn_s = slot_attention_s

        # Cross-view alignment module used during training.
        self.cross_matcher = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, 1)
        )

        self.slot_decoder = PatchDecoderCNN(
            object_dim=slot_dim,
            output_dim=slot_dim,
            grid_size=(32, 32),
            decoder_input_dim=256,
        )

        self.vlad = SlotSuperVLAD(slot_dim=slot_dim, desc_dim=vlad_dim)
        # self.vlad_s = SlotSuperVLAD(slot_dim=slot_dim, desc_dim=vlad_dim)

    def forward(self, feats_g=None, feats_s=None):
        """
        Training uses forward(g_feats, s_feats).
        Inference uses forward(g_feats).
        """
        # Inference with one input view.
        if not self.training:
            if feats_g is not None:
                slots = self.slot_attn_g(feats_g.flatten(start_dim=-2).transpose(1, 2))       # [B, K, D]
            else:
                slots = self.slot_attn_s(feats_s.flatten(start_dim=-2).transpose(1, 2))       # [B, K, D]
            B, K, _ = slots.shape
            keep_mask = torch.ones(B, K, device=slots.device)
            desc = self.vlad(slots, keep_mask)    # SuperVLAD
            return desc
        else:
            # Training with cross-view consistency supervision.
            slots_g = self.slot_attn_g(feats_g.flatten(start_dim=-2).transpose(1, 2))
            slots_s = self.slot_attn_s(feats_s.flatten(start_dim=-2).transpose(1, 2))

            B, K, _ = slots_g.shape

            # Construct pair features.
            sg = slots_g.unsqueeze(2).expand(-1, -1, K, -1)
            ss = slots_s.unsqueeze(1).expand(-1, K, -1, -1)
            pair_feats = torch.cat([sg, ss], dim=-1)
            similarity_logits = self.cross_matcher(pair_feats).squeeze(-1)  # [B, K, K]

            # Bidirectional Gumbel matching during training.
            dummy = torch.full((B, K, 1), -10.0, device=slots_g.device)
            logits_g2s = torch.cat([similarity_logits, dummy], dim=2)
            logits_s2g = torch.cat([similarity_logits.transpose(1, 2), dummy], dim=2)

            match_g2s, _ = gumbel_softmax(logits_g2s, self.temperature)
            match_s2g, _ = gumbel_softmax(logits_s2g, self.temperature)

            keep_mask_g = 1.0 - match_g2s[..., -1]   # [B,K]
            keep_mask_s = 1.0 - match_s2g[..., -1]

            # VLAD aggregation
            desc_g = self.vlad(slots_g, keep_mask_g)
            desc_s = self.vlad(slots_s, keep_mask_s)

            # Reconstruction
            recon_g = self.slot_decoder(slots_g, keep_mask_g)
            recon_s = self.slot_decoder(slots_s, keep_mask_s)

            aux_loss = keep_mask_g.sum() + keep_mask_s.sum()

            return desc_g, desc_s, recon_g, recon_s, aux_loss
