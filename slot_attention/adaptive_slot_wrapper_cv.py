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
    

class SlotSuperVLAD(nn.Module):
    """
    输入：经过跨视角匹配筛选后的 slots → 输出固定 3072 维全局描述子
    """
    def __init__(
        self,
        slot_dim: int = 256,
        num_clusters: int = 4,           # 64 是经典值 → 64×48=3072
        desc_dim: int = 3072,
        ghost_norm: bool = True,          # SuperVLAD 的关键 trick
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.ghost_norm = ghost_norm

        # 可学习聚类中心（就是 NetVLAD 的 codebook）
        self.centroids = nn.Parameter(torch.randn(num_clusters, slot_dim))

        # 先把 slot 投射到残差空间（SuperVLAD 推荐）
        self.project = nn.Linear(slot_dim, slot_dim, bias=False)
        nn.init.orthogonal_(self.project.weight)  # 关键初始化

        # 最终输出维度调整（如果 slot_dim=256 → 64×256=16384 太大，可以先降到 48）
        final_cluster_dim = desc_dim // num_clusters
        self.final_proj = nn.Linear(slot_dim, final_cluster_dim) if final_cluster_dim != slot_dim else nn.Identity()

    def forward(self, slots: torch.Tensor, keep_mask: torch.Tensor):
        """
        slots:      [B, K, D]
        keep_mask:  [B, K]  float, 1=跨视角匹配成功
        返回:       [B, 3072]  L2-normed
        """
        B, _, D = slots.shape

        # 1. 只保留匹配成功的 slots
        valid_slots = slots[keep_mask > 0.5]          # [N_valid, D]
        if valid_slots.shape[0] == 0:                 # 极端情况：一个都没匹配上
            return torch.zeros(B, self.num_clusters * (D if hasattr(self.final_proj, 'out_features') else D),
                              device=slots.device)

        # 2. 投影到残差空间
        residuals = self.final_proj(self.project(valid_slots))   # [N_valid, final_dim]

        # 3. 计算 soft assignment（比 hard 更平滑）
        dist = torch.cdist(residuals, self.centroids)            # [N_valid, C]
        soft_assign = F.softmax(-dist * 10.0, dim=-1)            # τ=0.1

        # 4. VLAD 聚合（每个 cluster 一个残差向量）
        vlad = torch.zeros(B, self.num_clusters, residuals.shape[-1], device=slots.device)
        batch_idx = (keep_mask > 0.5).nonzero(as_tuple=True)[0]

        # 累加残差
        for c in range(self.num_clusters):
            res_c = residuals - self.centroids[c]
            weighted = soft_assign[:, c:c+1] * res_c              # [N_valid, dim]
            vlad[batch_idx] += weighted.unsqueeze(0).sum(dim=1)  # 按 batch 累加

        # 5. SuperVLAD归一化
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

        # 双视角对齐模块（训练用）
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
        训练： forward(g_feats, s_feats)
        推理： forward(g_feats)
        """
        # ===== 推理模式：只输入一个视角 =====
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
            # ===== 训练模式：双视角一致性监督 =====
            slots_g = self.slot_attn_g(feats_g.flatten(start_dim=-2).transpose(1, 2))
            slots_s = self.slot_attn_s(feats_s.flatten(start_dim=-2).transpose(1, 2))

            B, K, _ = slots_g.shape

            # 构造 pair feature
            sg = slots_g.unsqueeze(2).expand(-1, -1, K, -1)
            ss = slots_s.unsqueeze(1).expand(-1, K, -1, -1)
            pair_feats = torch.cat([sg, ss], dim=-1)
            similarity_logits = self.cross_matcher(pair_feats).squeeze(-1)  # [B, K, K]

            # 双向 gumbel matching（训练）
            dummy = torch.full((B, K, 1), -10.0, device=slots_g.device)
            logits_g2s = torch.cat([similarity_logits, dummy], dim=2)
            logits_s2g = torch.cat([similarity_logits.transpose(1, 2), dummy], dim=2)

            match_g2s, _ = gumbel_softmax(logits_g2s, self.temperature)
            match_s2g, _ = gumbel_softmax(logits_s2g, self.temperature)

            keep_mask_g = 1.0 - match_g2s[..., -1]   # [B,K]
            keep_mask_s = 1.0 - match_s2g[..., -1]

            # VLAD 聚合
            desc_g = self.vlad(slots_g, keep_mask_g)
            desc_s = self.vlad(slots_s, keep_mask_s)

            # 重建
            recon_g = self.slot_decoder(slots_g, keep_mask_g)
            recon_s = self.slot_decoder(slots_s, keep_mask_s)

            aux_loss = keep_mask_g.sum() + keep_mask_s.sum()

            return desc_g, desc_s, recon_g, recon_s, aux_loss
