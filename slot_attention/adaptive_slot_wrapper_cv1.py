from __future__ import annotations

import math

import numpy as np

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
    """原始 Slot Attention 论文解码器：从 1×1 上采样到 32×32（正好 5 层 stride=2）"""
    return nn.Sequential(
        # 输入: [B, C, 1, 1]
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

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)

class SloteMixVPR(nn.Module):
    def __init__(self,
                 slot_dim=256,     # D
                 num_slots=20,     # K
                 out_channels=768, # channel proj dimension
                 out_rows=4,       # row proj → 4 rows
                 mix_depth=1,
                 mlp_ratio=1):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Token mixing across K (slot dimension)
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=num_slots, mlp_ratio=mlp_ratio)
            for _ in range(mix_depth)
        ])

        # Projections
        self.channel_proj = nn.Linear(slot_dim, out_channels)
        self.row_proj = nn.Linear(num_slots, out_rows)

    def forward(self, slots, keep_mask=None):
        """
        slots: B × K × D
        keep_mask: B × K (0/1)
        """
        # --- (1) Apply keep_mask ---
        if keep_mask is not None:
            keep_mask = keep_mask.unsqueeze(-1)        # B × K × 1
            slots = slots * keep_mask                  # masked slots
        
        # --- (2) Mix tokens along K ---
        # x: B × D × K
        x = slots.permute(0, 2, 1)
        x = self.mix(x)                                # mixer

        # --- (3) channel projection (per slot) ---
        x = x.permute(0, 2, 1)                         # B × K × D
        x = self.channel_proj(x)                       # B × K × out_channels
        
        # --- (4) row projection across slots ---
        x = x.permute(0, 2, 1)                         # B × out_channels × K
        x = self.row_proj(x)                           # B × out_channels × out_rows

        # --- (5) flatten + L2 norm ---
        x = F.normalize(x.flatten(1), p=2, dim=-1)     # B × (out_rows*out_channels)

        return x

class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 flatten_dim=1536,
                 in_h=32,
                 in_w=32,
                 out_channels=512,
                 mix_depth=2,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()
        # self.in_h = in_h
        # self.in_w = in_w
        self.in_channels = in_channels  # depth of input feature maps 特征图通道数

        self.out_channels = out_channels  # depth wise projection dimension 深度投影尺寸
        self.out_rows = out_rows  # row wise projection dimesion 列投影尺寸

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers //Mixer的数量
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block //Mixer块的中间投影层的比率

        hw = in_h * in_w
        # hw = flatten_dim
        # 定义一个Sequential容器，用于叠加FeatureMixerLayer
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)  # hw输入尺寸，out_rows输出尺寸


    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)  # Feature-Mixer模块
        x = x.permute(0, 2, 1)  # 将数据转换成 0块 2行1列
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)  # 将数据转换成 0块 2行1列
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)  # 将x展平,并正则化
        return x
    
# -------------------------
# Matching-based CVGL module (Version C)
# -------------------------
class CVGLSlotVLAD(nn.Module):
    def __init__(
        self,
        slot_attention_g,
        slot_attention_s=None,
        slot_dim=768,
        num_slots=32,
        temperature=0.1,
        keep_thres: float = 0.2,   # optional threshold for hard keep in certain modes
        topk: int = None           # optional topk filter for matches (per-row)
    ):
        """
        slot_attention_g : callable module mapping [B, Npatches, C] -> [B, K, slot_dim]
        slot_attention_s : if None and share_slot_attn True, uses slot_attention_g
        share_slot_attn  : whether to share slot attention weights
        temperature      : temperature for softmax similarity
        topk             : if int, only keep topk partners per slot when computing context (reduces mem)
        """
        super().__init__()
        self.temperature = temperature
        self.slot_attn_g = slot_attention_g
        self.slot_attn_s = slot_attention_s

        self.slot_dim = slot_dim
        self.topk = topk
        self.keep_thres = keep_thres

        # cross_refiner: takes slot and weighted partner context -> outputs keep prob
        self.cross_refiner = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, 1)
        )

        # lightweight similarity refinement (optional) - small MLP on pair concat to refine sim
        self.pair_refine = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, 1)
        )

        self.slot_decoder = PatchDecoderCNN(
            object_dim=slot_dim,
            output_dim=slot_dim,
            grid_size=(32, 32),
            decoder_input_dim=768,
        )

        # self.vlad = SloteMixVPR(slot_dim=slot_dim, num_slots=num_slots)
        self.vlad = MixVPR(in_channels=self.slot_dim, out_channels=1024)

    def _compute_similarity(self, slots_g, slots_s):
        """
        slots_g: [B, Kg, D]
        slots_s: [B, Ks, D]
        returns:
            sim: [B, Kg, Ks] (raw)
        """
        D = slots_g.shape[-1]
        # normalized dot product (scaled)
        ng = F.normalize(slots_g, p=2, dim=-1)  # [B,Kg,D]
        ns = F.normalize(slots_s, p=2, dim=-1)  # [B,Ks,D]
        sim = torch.matmul(ng, ns.transpose(-1, -2))  # [B,Kg,Ks]
        sim = sim / max(math.sqrt(D), 1.0)
        return sim

    def _row_col_softmax(self, sim):
        """
        compute row-softmax and col-softmax and combine as symmetric confidence
        sim: [B, Kg, Ks]
        returns:
            row_soft: [B, Kg, Ks]
            col_soft: [B, Kg, Ks]
            combined: [B, Kg, Ks]  (elementwise product)
        """
        row_soft = F.softmax(sim / self.temperature, dim=-1)
        col_soft = F.softmax(sim.transpose(-1, -2) / self.temperature, dim=-1).transpose(-1, -2)
        combined = row_soft * col_soft
        return row_soft, col_soft, combined

    def _topk_mask(self, combined, k):
        """
        keep only top-k per row in combined, zero others (soft weights preserved)
        combined: [B, Kg, Ks]
        returns masked combined
        """
        if k is None:
            return combined
        B, Kg, Ks = combined.shape
        topk_vals, topk_idx = combined.topk(min(k, Ks), dim=-1)  # [B,Kg,k]
        mask = torch.zeros_like(combined)
        # scatter 1s to positions; vectorized:
        batch_idx = torch.arange(B, device=combined.device)[:, None, None]
        row_idx = torch.arange(Kg, device=combined.device)[None, :, None]
        col_idx = topk_idx
        mask[batch_idx, row_idx, col_idx] = 1.0
        return combined * mask

    def _mutual_nearest(self, sim):
        """
        mutual nearest: boolean matrix where mutual argmax holds.
        sim: [B,Kg,Ks]
        returns:
            mutual_mask: [B,Kg,Ks] floats 0/1
        """
        argmax_s = sim.argmax(dim=-1)      # [B,Kg] : for each g -> index of best s
        argmax_g = sim.argmax(dim=-2)      # [B,Ks] : for each s -> index of best g

        B, Kg, Ks = sim.shape
        # make mutual mask
        g_idx = torch.arange(Kg, device=sim.device)[None, :, None]  # [1,Kg,1]
        b_idx = torch.arange(B, device=sim.device)[:, None, None]   # [B,1,1]

        # argmax_s[b,i] gives j; we want to check argmax_g[b,j] == i
        mutual = torch.zeros_like(sim)
        # vectorized check:
        j = argmax_s  # [B,Kg]
        # gather argmax_g[b, j] -> [B,Kg]
        argmax_g_at_j = argmax_g.gather(1, j)  # careful shapes: argmax_g [B, Ks] ; gather with dim=1
        # but need to align dims — implement with loop-free indexing using advanced indexing:
        # build indices for gather:
        # argmax_g_at_j[b,i] = argmax_g[b, argmax_s[b,i]]
        # implement with torch.take_along_dim
        argmax_g_at_j = torch.take_along_dim(argmax_g.unsqueeze(1), j.unsqueeze(1), dim=2).squeeze(1)  # [B,Kg]
        mutual_indices = (argmax_g_at_j == torch.arange(Kg, device=sim.device)[None, :])  # error-prone to build; simpler approach below

        # Simpler robust approach: compute mutual using argmax per-axis with loops over batch (safe and small K)
        # K is typically small (e.g., 16-64); using a small loop over batch is acceptable
        mutual = torch.zeros_like(sim)
        for b in range(B):
            ag = argmax_s[b]  # [Kg]
            agrev = argmax_g[b]  # [Ks]
            for i in range(Kg):
                jidx = int(ag[i].item())
                if int(agrev[jidx].item()) == i:
                    mutual[b, i, jidx] = 1.0
        return mutual

    def forward(self, feats_g=None, feats_s=None):
        """
        Training:
            provide feats_g and feats_s -> does matching -> returns desc_g, desc_s, recon_g, recon_s, aux_loss
        Inference:
            provide only feats_g (or feats_s) -> compute slots -> VLAD with keep_mask=1
        """
        # ---- inference (eval) ----
        if not self.training:
            if feats_g is not None:
                slots = self.slot_attn_g(feats_g.flatten(start_dim=-2).transpose(1, 2))  # [B, K, D]
            else:
                slots = self.slot_attn_s(feats_s.flatten(start_dim=-2).transpose(1, 2))
            B, K, _ = slots.shape
            keep_mask = torch.ones(B, K, device=slots.device)
            recon = self.slot_decoder(slots, keep_mask)
            desc = self.vlad(recon.permute(0, 2, 1))
            return desc

        # ---- training ----
        slots_g = self.slot_attn_g(feats_g.flatten(start_dim=-2).transpose(1, 2))  # [B, Kg, D]
        slots_s = self.slot_attn_s(feats_s.flatten(start_dim=-2).transpose(1, 2))  # [B, Ks, D]

        B, _, _ = slots_g.shape

        # 1) compute raw similarity
        sim_raw = self._compute_similarity(slots_g, slots_s)  # [B, Kg, Ks]

        # 2) optional pair-wise refinement (cheap): run small MLP on concatenated pairs (applied selectively)
        # For memory, avoid full Kg*Ks MLP unless K small. We'll refine sim for top candidates using row soft.
        _, _, combined = self._row_col_softmax(sim_raw)

        # optional topk sparsification to reduce noise
        combined = self._topk_mask(combined, self.topk)  # [B,Kg,Ks]

        # 3) construct partner context for each slot (weighted sum of partners)
        # For g: context_g = combined @ slots_s  -> [B, Kg, D]
        context_g = torch.matmul(combined, slots_s)  # [B, Kg, D]
        # For s: context_s = combined.transpose(1,2) @ slots_g -> [B, Ks, D]
        context_s = torch.matmul(combined.transpose(-1, -2), slots_g)  # [B, Ks, D]

        # 4) refine keep probability via cross_refiner on (slot, context)
        concat_g = torch.cat([slots_g, context_g], dim=-1)  # [B, Kg, 2D]
        concat_s = torch.cat([slots_s, context_s], dim=-1)  # [B, Ks, 2D]
        keep_logits_g = self.cross_refiner(concat_g).squeeze(-1)  # [B, Kg]
        keep_logits_s = self.cross_refiner(concat_s).squeeze(-1)  # [B, Ks]
        keep_prob_g = torch.sigmoid(keep_logits_g)  # continuous [0,1]
        keep_prob_s = torch.sigmoid(keep_logits_s)

        # optionally enforce mutual constraint: boost keep for mutually matched slots
        # mutual_mask: [B, Kg, Ks] {0,1} using mutual nearest (safe small loop)
        mutual_mask = self._mutual_nearest(sim_raw)  # [B, Kg, Ks]
        # compute if a g slot has any mutual partner
        g_has_mutual = (mutual_mask.sum(dim=-1) > 0).float()  # [B, Kg]
        s_has_mutual = (mutual_mask.sum(dim=-2) > 0).float()  # [B, Ks]
        # combine signals: if has mutual then encourage keep_prob -> multiply factor
        keep_prob_g = torch.clamp(keep_prob_g + 0.5 * g_has_mutual, 0.0, 1.0)
        keep_prob_s = torch.clamp(keep_prob_s + 0.5 * s_has_mutual, 0.0, 1.0)

        # 5) produce final keep_mask (soft). For some losses you may want hard mask:
        keep_mask_g = keep_prob_g  # [B, Kg]
        keep_mask_s = keep_prob_s  # [B, Ks]

        # 6) reconstruction via decoder (decoder can accept mask as gating)
        recon_g = self.slot_decoder(slots_g, keep_mask_g)
        recon_s = self.slot_decoder(slots_s, keep_mask_s)

        # 7) VLAD aggregation using shared vlad
        desc_g = self.vlad(recon_g.permute(0, 2, 1))
        desc_s = self.vlad(recon_s.permute(0, 2, 1))

        # 8) auxiliary losses:
        # - coverage loss: prefer some slots kept (avoid trivial all-zero) -> encourage keep sum
        # - mutual agreement loss: encourage matched pairs to have high sim
        coverage_loss = -(keep_mask_g.mean() + keep_mask_s.mean())  # negative to encourage keeping (tunable)
        # mutual agreement: encourage combined scores to be high on mutual pairs
        if mutual_mask.sum() > 0:
            mutual_scores = (combined * mutual_mask).sum() / (mutual_mask.sum() + 1e-6)
            mutual_loss = -torch.log(mutual_scores + 1e-6)
        else:
            mutual_loss = torch.tensor(0.0, device=desc_g.device)

        aux_loss = coverage_loss + mutual_loss

        return desc_g, desc_s, recon_g, recon_s, slots_g, slots_s, aux_loss
