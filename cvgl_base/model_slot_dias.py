import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvgl_base import helper
from einops import rearrange

# from .point_pn.point_pn import Point_PN

from slot_attention.dias_slot_attention import MLP, NormalShared, LearntPositionalEmbedding
from slot_attention.dias_slot_wrapper_cv import DIAS, SlotAttentionWithAllAttent, ARRandTransformerDecoder


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

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


class SlotMixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        # x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x

class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 主干网络
                 model_name='dinov2_vitb14_MixVPR',
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,
                 ):
        super().__init__()
        self.pretrained = pretrained  # 是否预训练
        self.layers_to_freeze = layers_to_freeze  # 冻结网络层名称
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name

        # ----------------------------------
        # get the backbone and the aggregator 获得主干网络和聚合器
        self.backbone = helper.get_backbone(backbone_arch=backbone_arch, pretrained=pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs)
        # self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
        # self.proj = nn.Sequential(
        #     nn.Linear(4096, 5120),
        #     nn.ReLU()
        # )

    # the forward pass of the lightning model
    def forward(self, x):   # x: [B, 3, 448, 448]
        x, _, _ = self.backbone(x)    # x: [B, 1024, 32, 32]
        # x = self.aggregator(x)
        # x = self.proj(x)
        return x

class SlotMixFusion(nn.Module):
    def __init__(self, in_channels, slot_channels=None, num_slots=8, alpha=0.8, slot_proj_dim=None):
        super().__init__()
        self.in_channels = in_channels  # C
        self.num_slots = num_slots
        self.alpha = alpha
        slot_channels = slot_channels
        slot_proj_dim = slot_proj_dim

        # project slot pooled vector -> channel dim for FiLM / add
        self.slot_to_channel = nn.Linear(slot_channels, slot_proj_dim)
        # small MLP to compute gating per slot (per-channel)
        self.slot_gate = nn.Sequential(
            nn.Linear(slot_proj_dim, slot_proj_dim//2),
            nn.GELU(),
            nn.Linear(slot_proj_dim//2, slot_proj_dim),  # produce channel gate
        )
        # optional residual conv to mix spatially after fusion
        # self.res_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        # self.norm = nn.LayerNorm(in_channels)

    def forward(self, feat, slots):
        """
        feat: B, C, H, W  (dense feature map from backbone)
        slots: B, K, H, W (slot spatial maps / masks)  -- assumed same H,W
        returns: feat_enhanced: B, C, H, W
        """
        B, C, H, W = feat.shape
        slots = slots.view(slots.size(0), slots.size(1), H, W)
        B2, K, H2, W2 = slots.shape
        assert B==B2 and H==H2 and W==W2, "feat and slots must share batch and spatial dims"

        # 1) compute slot vectors by spatially pooling feature weighted by the slot map
        # slots may not be normalized; softmax over spatial to get attention
        slots_flat = slots.reshape(B, K, -1)  # B,K,H*W
        attn = F.softmax(slots_flat, dim=-1)  # B, K, 1, H*W
        feat_flat = feat.reshape(B, C, -1)  # B, C, H*W

        # slot_vec: B, K, C  (weighted sum of features under each slot map)
        slot_vec = torch.matmul(attn, feat_flat.permute(0,2,1))  # (B,K,1,C)
        # slot_vec = slot_vec.squeeze(2)  # B,K,C

        # 2) project slot_vec -> channel modulation
        # option A: compute per-slot channel gates then aggregate across slots
        slot_proj = self.slot_to_channel(slot_vec)  # B,K,C
        # compute gate per slot: B,K,C -> merge K via softmax-weighted sum by slot total mass
        # measure slot strength by sum over spatial (before softmax)
        slot_strength = slots_flat.sum(-1)  # B,K
        slot_alpha = F.softmax(slot_strength, dim=-1).unsqueeze(-1)  # B,K,1

        # gated channel vector: weighted sum over slots
        channel_vec = (slot_proj * slot_alpha).sum(dim=1)  # B, C

        # 3) FiLM style: produce gamma & beta from channel_vec
        gamma = torch.sigmoid(self.slot_gate(channel_vec))  # B, C

        # apply FiLM to feat
        gamma = gamma.view(B, C, 1, 1)
        feat_mod = feat * (1.0 + gamma)  # residual scaling

        # 4) spatial-aware augmentation: reconstruct slot maps into channel space and add
        # project each slot vector to C, expand spatially and combine via slot maps as weights
        slot_to_channel = slot_proj  # B,K,C
        # normalize slot maps along K so they act as convex weights per-pixel
        slot_maps = F.softmax(slots, dim=1)  # B,K,H,W
        # compute per-pixel aggregated vector: sum_k slot_map_k * slot_vec_k
        # reshape for broadcasting
        slot_to_channel = slot_to_channel.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)  # B,C,K,1,1
        slot_maps = slot_maps.unsqueeze(1)  # B,1,K,H,W
        slot_spatial_add = (slot_maps * slot_to_channel).sum(dim=2)  # B,C,H,W

        # combine with feat_mod
        feat_enh = feat_mod + slot_spatial_add
        feat_enh = rearrange(feat_enh, 'b c h w -> b c (h w)')

        # optional conv + norm + residual
        # feat_enh = feat_enh.permute(0, 2, 1)
        # feat_enh = self.res_conv(feat_enh)
        # feat_enh = feat_enh.permute(0, 2, 1)
        # feat_enh = self.norm(feat_enh) + feat_enh  # residual
        
        feat_out = self.alpha*feat_enh + (1.0-self.alpha)*feat_flat

        return feat_out

class SlotCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.ca = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)

    def forward(self, att_query, att_context):
        """
        att_query:   当前视角 slots  [B, K, D] → Q
        att_context: 另一视角 slots  [B, K, D] → K,V
        """
        # 转换为 [K, B, D]
        q = att_query.transpose(0, 1)
        k = att_context.transpose(0, 1)
        v = att_context.transpose(0, 1)

        out, _ = self.ca(q, k, v)     # Q=当前视角，KV=另一个视角

        # 回到 [B, K, D]
        out = out.transpose(0, 1)

        # 残差增强 slot 表达
        return att_query + out

class SlotRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()             # 限制权重到 [0,1]
        )

    def forward(self, slots):         # [B, K, D]
        return self.net(slots)        # [B, K, 1]

class SlotCrossMoE(nn.Module):
    def __init__(self, dim, nheads=4):
        super().__init__()
        self.ca12 = SlotCrossAttention(dim, nheads)   # 1 ← 2
        self.ca21 = SlotCrossAttention(dim, nheads)   # 2 ← 1
        self.router = SlotRouter(dim)

    def forward(self, att1, att2):
        # cross attention (单query版本)
        f1 = self.ca12(att1, att2)     # att1 query 视角1从视角2抽取信息
        f2 = self.ca21(att2, att1)     # att2 query 视角2从视角1抽取信息

        # MoE Router
        w1 = self.router(f1)           # [B, K, 1]
        w2 = self.router(f2)

        # 加权（slot gating）
        att1_w = att1 * w1
        att2_w = att2 * w2

        return att1_w, att2_w

class TimmModel_slot(nn.Module):

    def __init__(self,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 vfm_dim = 768,
                 emb_dim = 1024,
                 num_slots = 16,
                 iters = 3,
                 layer1=8,
                 alpha=0.8,
                 ):

        super(TimmModel_slot, self).__init__()

        self.img_size = img_size
        self.alpha = alpha
        self.vfm_dim = vfm_dim
        self.emb_dim = emb_dim
        self.num_slots = num_slots
        self.iters = iters
        
        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, layer1=layer1)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Slot Attention
        encode_posit_embed = nn.Identity()
        encode_project = MLP(in_dim=self.vfm_dim, dims=[self.vfm_dim, self.vfm_dim], ln="pre")

        initializ = NormalShared(num=self.num_slots, dim=self.emb_dim)

        aggregat = SlotAttentionWithAllAttent(num_iter=self.iters, embed_dim=self.emb_dim, ffn_dim=self.emb_dim*4, kv_dim=self.vfm_dim, trunc_bp=None)

        # wrap the slot attention
        project1 = nn.Sequential(
            nn.Linear(in_features=self.vfm_dim, out_features=self.vfm_dim, bias=False),
            nn.LayerNorm(normalized_shape=self.vfm_dim)
        )
        project2 = nn.Sequential(
            nn.Linear(in_features=self.emb_dim, out_features=self.vfm_dim, bias=False),
            nn.LayerNorm(normalized_shape=self.vfm_dim)
        )
        backbone = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.vfm_dim,
                nhead=4,
                dim_feedforward=self.vfm_dim*2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False
            ),
            num_layers=4
        )
        readout = nn.Identity()

        decoder = ARRandTransformerDecoder(vfm_dim=self.vfm_dim,
                                        posit_embed=LearntPositionalEmbedding(resolut=[32*32], embed_dim=self.vfm_dim),
                                        project1=project1,
                                        project2=project2,
                                        backbone=backbone,
                                        readout=readout)

        self.slot_model = DIAS(
            encode_posit_embed=encode_posit_embed,
            encode_project=encode_project,
            initializ=initializ,
            aggregat=aggregat,
            decode=decoder
        )

        self.mix_fusion = SlotMixFusion(
            in_channels=self.emb_dim,
            slot_channels=self.vfm_dim,
            slot_proj_dim=self.vfm_dim,
            num_slots=self.num_slots,
            alpha=self.alpha
        )

        self.slot_mixvpr = SlotMixVPR(
            in_channels=self.vfm_dim,
            in_h=32,
            in_w=32,
            mix_depth=2,
            out_channels=self.emb_dim
        )

        self.slot_moe = SlotCrossMoE(
            dim=self.emb_dim,
            nheads=4
        )

        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)

    def get_config(self):
        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):
        if self.training:
            feats1 = self.model(img1) # [B, P, D]
            feats2 = self.model(img2) # [B, P, D]

            # Agg
            feats1_flat = rearrange(feats1, 'b p h w -> b p (h w)')
            des_g = self.slot_mixvpr(feats1_flat)
            feats2_flat = rearrange(feats2, 'b p h w -> b p (h w)')
            des_s = self.slot_mixvpr(feats2_flat)

            # Slot Attention            
            _, _, _, attent1, recon1 = self.slot_model(feats1)
            _, _, _, attent2, recon2 = self.slot_model(feats2)

            # Cross-view MoE
            attent1 = rearrange(attent1, 'b k h w -> b k (h w)')
            attent2 = rearrange(attent2, 'b k h w -> b k (h w)')
            attent1, attent2 = self.slot_moe(attent1, attent2)

            mix1 = self.mix_fusion(feats1, attent1) # [B, P, D]
            mix2 = self.mix_fusion(feats2, attent2) # [B, P, D]

            des1 = self.slot_mixvpr(mix1) # [B, D1]
            des2 = self.slot_mixvpr(mix2) # [B, D1]

            return (feats1, feats2), (des_g, des_s, des1, des2), (attent1, attent2), (recon1, recon2)
        else:
            feats1 = self.model(img1)
            # For distillation only
            _, _, _, attent1, _ = self.slot_model(feats1)
            # attent1 = rearrange(attent1, 'b k h w -> b k (h w)')

            # # Intra-view MoE
            # attent1, _ = self.slot_moe(attent1, attent1)

            mix1 = self.mix_fusion(feats1, attent1)
            # feats1_flat = rearrange(mix1, 'b p h w -> b p (h w)')
            des1 = self.slot_mixvpr(mix1)

            return des1

class TimmModel_base(nn.Module):

    def __init__(self,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 vfm_dim = 768,
                 emb_dim = 1024,
                 num_slots = 16,
                 iters = 3,
                 layer1=8,
                 alpha=0.8,
                 ):

        super(TimmModel_base, self).__init__()

        self.img_size = img_size
        self.alpha = alpha
        self.vfm_dim = vfm_dim
        self.emb_dim = emb_dim
        self.num_slots = num_slots
        self.iters = iters
        
        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, layer1=layer1)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.slot_mixvpr = SlotMixVPR(
            in_channels=self.vfm_dim,
            in_h=32,
            in_w=32,
            mix_depth=2,
            out_channels=self.emb_dim
        )

        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)

    def get_config(self):
        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):
        if self.training:
            feats1 = self.model(img1) # [B, P, D]
            feats2 = self.model(img2) # [B, P, D]

            # Agg
            feats1_flat = rearrange(feats1, 'b p h w -> b p (h w)')
            des_g = self.slot_mixvpr(feats1_flat)
            feats2_flat = rearrange(feats2, 'b p h w -> b p (h w)')
            des_s = self.slot_mixvpr(feats2_flat)

            return des_g, des_s
        else:
            feats1 = self.model(img1)
            # # For distillation only
            # _, _, _, attent1, _ = self.slot_model(feats1)
            # attent1 = rearrange(attent1, 'b k h w -> b k (h w)')

            # # Intra-view MoE
            # # attent1, _ = self.slot_moe(attent1, attent1)

            # mix1 = self.mix_fusion(feats1, attent1)
            feats1_flat = rearrange(feats1, 'b p h w -> b p (h w)')
            des1 = self.slot_mixvpr(feats1_flat)

            return des1