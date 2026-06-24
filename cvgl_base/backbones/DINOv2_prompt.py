# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat
from typing import Type, Optional


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitl14_reg", "dinov2_vitg14", "dinov3_vitb14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class LightweightQueryFormer(nn.Module):
    """
    Simplified Lightweight QueryFormer (Gated Attention)
    - 输入: patch tokens [B, F, C]
    - 输出: prompts [B, N, C]
    """
    def __init__(self, embed_dim, num_prompts=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_prompts = num_prompts
        self.query_tokens = nn.Parameter(torch.randn(num_prompts, embed_dim))  # 可学习 query

        # gating mechanism
        self.gate_proj = nn.Linear(embed_dim, embed_dim)

        # FFN (轻量 MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        """
        B, F, C = patch_tokens.shape
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # gated cross-attention (简化版)
        # score = (Q * Gate(Q)) @ P^T
        q_proj = self.gate_proj(self.norm1(queries))   # [B, N, C]
        attn_logits = torch.einsum('bnc,bfc->bnf', q_proj, patch_tokens) / (C ** 0.5)  # [B, N, F]
        attn = torch.softmax(attn_logits, dim=-1)

        # 聚合 patch tokens
        attn_output = torch.matmul(attn, patch_tokens)  # [B, N, C]

        # 残差 + FFN
        x = queries + attn_output
        x = x + self.ffn(self.norm2(x))
        return x  # [B, N, C]

class MLP_PromptGenerator(nn.Module):
    """
    2025风格: 用MLP非线性映射生成prompts
    参考 LLaVA-Next, Qwen-VL-2 的视觉prompt做法
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2  # 常见设置: 扩大一倍

        # token → prompt 映射
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # 或 SwiGLU 更前沿
        #     nn.Linear(hidden_dim, num_prompts * embed_dim)
        # )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            # nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),   # 或 SwiGLU 更前沿
            nn.Linear(1024, num_prompts)
        )
        # self.mlp = nn.Sequential(
        #     # nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # 或 SwiGLU 更前沿
        #     nn.Linear(787, num_prompts)
        # )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        输出: prompts [B, num_prompts, C]
        """
        B, F, C = patch_tokens.shape

        # step1: 先对所有patch求平均 (更稳定)，也可换成 learnable pooling
        patch_tokens = self.layer_norm(patch_tokens)
        pooled = self.mlp(patch_tokens.permute(0, 2, 1))  # [B, C]

        # step2: MLP 非线性投影
        # prompts = self.mlp(pooled)  # [B, num_prompts*C]

        # step3: reshape
        # prompts = prompts.view(B, self.num_prompts, C)
        prompts = pooled.permute(0, 2, 1)

        return prompts


class MLP_PromptGenerator_(nn.Module):
    """
    2025风格: 用MLP非线性映射生成prompts
    参考 LLaVA-Next, Qwen-VL-2 的视觉prompt做法
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2  # 常见设置: 扩大一倍

        # token → prompt 映射
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # 或 SwiGLU 更前沿
        #     nn.Linear(hidden_dim, num_prompts * embed_dim)
        # )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(1024, num_prompts)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        输出: prompts [B, num_prompts, C]
        """
        B, F, C = patch_tokens.shape

        # step1: 先对所有patch求平均 (更稳定)，也可换成 learnable pooling
        patch_tokens = self.layer_norm(patch_tokens)
        pooled = self.mlp(patch_tokens.permute(0, 2, 1))  # [B, C]

        # step2: MLP 非线性投影
        # prompts = self.mlp(pooled)  # [B, num_prompts*C]

        # step3: reshape
        # prompts = prompts.view(B, self.num_prompts, C)
        prompts = pooled.permute(0, 2, 1)

        return prompts


class MLP_TopKPromptGenerator(nn.Module):
    """
    Top-k Prompt Generator:
    - 直接选取 top-k = num_prompts 个 patch tokens 作为候选（按 learnable score）
    - 对每个被选 token 应用 MLP 映射，得到对应的 prompt embedding
    - 若 num_patches < num_prompts，则用已选 prompts 的均值填充
    输出: prompts [B, num_prompts, C]
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2

        # token重要性评分（标量）
        self.score_proj = nn.Linear(embed_dim, 1)

        # token 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 每个被选 token 映射为 prompt 的 MLP（对最后一维进行映射）
        # 输入形状可以是 [B, k, C]，Linear 会应用到最后一维
        self.token_to_prompt = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, num_patches, C]
        return: prompts [B, num_prompts, C]
        """
        B, num_patches, C = patch_tokens.shape

        # 1) LayerNorm
        tokens = self.layer_norm(patch_tokens)  # [B, num_patches, C]

        # 2) 计算分数并选 top-k（k = num_prompts 或 num_patches if smaller）
        scores = self.score_proj(tokens).squeeze(-1)  # [B, num_patches]
        k = min(self.num_prompts, num_patches)
        _, topk_idx = torch.topk(scores, k, dim=1)  # [B, k]

        # 3) 根据索引提取 top-k tokens
        # 使用 gather：扩展索引以匹配最后一维
        topk_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))  # [B, k, C]

        # 4) 每个被选 token 映射为 prompt embedding（保持 per-token 多样性）
        prompts_k = self.token_to_prompt(topk_tokens)  # [B, k, C]

        # 5) 若 k < num_prompts（patch 太少），用均值填充剩余 slot
        if k < self.num_prompts:
            # 用已选 prompts 的均值作为填充（比 0 更稳定）
            filler = prompts_k.mean(dim=1, keepdim=True).expand(B, self.num_prompts - k, C)  # [B, num_prompts-k, C]
            prompts = torch.cat([prompts_k, filler], dim=1)  # [B, num_prompts, C]
        else:
            prompts = prompts_k  # 已经等于 num_prompts

        return prompts  # [B, num_prompts, C]


class ChannelPromptGenerator(nn.Module):
    """
    改进版：通道注意力 + MLP prompt 映射
    设计目标：得到每个通道的权重 (B, 1, C)，再将 (B, F, C) → (B, num_prompts, C)
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2

        # ---- 通道注意力生成器 (B, F, C) → (B, 1, C)
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        # ---- Prompt 生成 MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # ---- 融合层：F → num_prompts
        self.fusion = nn.Linear(1, num_prompts)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        输出: prompts [B, num_prompts, C]
        """
        B, F, C = patch_tokens.shape

        # ---- Step1: 通道注意力 (B, F, C) → (B, 1, C)
        # 对每个通道独立生成注意力
        channel_weights = self.channel_attn(patch_tokens.mean(dim=1, keepdim=True))  # [B, 1, C]

        # ---- Step2: 应用通道注意力
        pooled = channel_weights * patch_tokens  # [B, F, C]

        # ---- Step3: MLP 映射（逐 token）
        pooled = self.mlp(pooled)  # [B, F, C]

        # ---- Step4: 将 F 压缩到 num_prompts
        # 方法：沿 F 维取平均后再映射为 num_prompts
        pooled_mean = pooled.mean(dim=2, keepdim=True)  # [B, F, 1]
        weights = self.fusion(pooled_mean.transpose(1, 2))  # [B, num_prompts, F]
        attn = F.softmax(weights, dim=-1)  # [B, num_prompts, F]

        # ---- Step5: 聚合得到最终 prompts
        prompts = torch.bmm(attn, pooled)  # [B, num_prompts, C]

        return prompts

class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数，源自 PaLM 论文，在 Llama / Mistral 中广泛使用。
    比 GELU / ReLU 更具表现力。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 2 * D]
        输出: [..., D]
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class MLP(nn.Module):
    """
    一个更通用、可配置的前馈网络 (FFN) 模块。
    参考 ViT / Llama 等模型中的 FFN 设计。
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # SwiGLU 需要特殊处理，因为它的第一个线性层输出维度是两倍
        self.is_swiglu = act_layer is SwiGLU
        ffn_hidden_features = hidden_features * 2 if self.is_swiglu else hidden_features

        self.fc1 = nn.Linear(in_features, ffn_hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 2. 重构后的 PatchPromptGenerator ----

class PatchPromptGeneratorV2(nn.Module):
    """
    Patch-Level Attention Prompt Generator (V2 - 2025 Style)

    通过注意力机制将大量的图像 Patch 特征聚合成一小组高信息量的视觉 Prompt。
    设计灵感来源于最新的多模态大模型，如 LLaVA-Next, Qwen2-VL, InternVL2。

    改进点:
    - 使用可配置的 MLP 模块，轻松切换激活函数 (如 SwiGLU)。
    - 在最终的投影层加入了残差连接，增强了模型的稳定性和表现力。
    - 结构更清晰，配置更灵活（使用 mlp_ratio）。
    - 提供了完整的类型注解和文档。
    """
    def __init__(
        self,
        embed_dim: int,
        num_prompts: int = 8,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = SwiGLU, # 默认使用更先进的 SwiGLU
        add_residual: bool = True,
    ):
        """
        Args:
            embed_dim (int): 输入 Patch Token 的特征维度 (C)。
            num_prompts (int): 要生成的视觉 Prompt 的数量。
            mlp_ratio (float): MLP 中间隐藏层的维度相对于 embed_dim 的比例。
            act_layer (Type[nn.Module]): FFN 中使用的激活函数类。
            add_residual (bool): 是否在投影层后添加残差连接。
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.add_residual = add_residual
        hidden_dim = int(embed_dim * mlp_ratio)

        # ---- Patch Attention Scorer: 生成每个 patch 对各个 prompt 的注意力权重
        # LayerNorm -> MLP -> Output
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MLP(
                in_features=embed_dim,
                hidden_features=hidden_dim,
                out_features=num_prompts,
                act_layer=act_layer
            )
        )

        # ---- Prompt Projector: 对聚合后的特征进行 refine
        # LayerNorm -> MLP
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MLP(
                in_features=embed_dim,
                hidden_features=hidden_dim,
                out_features=embed_dim,
                act_layer=act_layer
            )
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features (torch.Tensor): 输入的图像 Patch 特征, shape: [B, F, C]
                                           B = Batch size
                                           F = Number of patches (e.g., 256 for 224x224 image)
                                           C = Embedding dimension

        Returns:
            torch.Tensor: 生成的视觉 prompts, shape: [B, num_prompts, C]
        """

        # Step 1: 计算每个 Patch 对每个 Prompt 的注意力分数
        # [B, F, C] -> [B, F, num_prompts]
        attention_scores = self.scorer(patch_features)

        # Step 2: 沿 Patch 维度 (F) 进行 softmax 归一化，得到权重
        # transpose: [B, num_prompts, F]
        # softmax:   [B, num_prompts, F]
        attention_weights = F.softmax(attention_scores.transpose(1, 2), dim=-1)

        # Step 3: 加权求和，将 F 个 Patch 特征聚合成 num_prompts 个向量
        # torch.bmm([B, num_prompts, F], [B, F, C]) -> [B, num_prompts, C]
        aggregated_prompts = torch.bmm(attention_weights, patch_features)

        # Step 4: 通过 Projector 进行特征变换和精炼 (Refinement)
        # [B, num_prompts, C] -> [B, num_prompts, C]
        refined_prompts = self.projector(aggregated_prompts)

        # Step 5 (Optional but Recommended): 添加残差连接
        if self.add_residual:
            prompts = aggregated_prompts + refined_prompts
        else:
            prompts = refined_prompts
            
        return prompts

class DinoV2_prompt(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, style_aligner, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.style_aligner = style_aligner
        self.model_name = model_name.lower()[:-7]  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('请确认layer的正确性！vitl14最高block层为23层')
        #         exit()
        elif 'dinov2'in self.model_name and 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'dinov3'in self.model_name and 'vitb16' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3_wt/dinov3_vitb16_pretrain_lvd1689m.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home/liuyh/GEO/code/CVCities-main/dinov2/dinov2_vits14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        # ---------------- visual style prompts ----------------
        embed_dim = self.dino_model.embed_dim
        self.num_prompts = 4
        self.prompt_gen = PatchPromptGeneratorV2(
            embed_dim=embed_dim,
            num_prompts=self.num_prompts,
        )


    def forward(self, x, style_x=None, masks=None):

        # x = self.dino_model.forward_features(x)

        # if isinstance(x, list):
        #     return self.forward_features_list(x, masks)
        
        if self.training:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)

            # 注意：保持原来代码的分片方式（cls 在 0），原作者用 x[:,2:,:] 作为 patch tokens
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C]  （与原代码一致）
            # 由 patch tokens 生成 prompts
            prompts = self.prompt_gen(patch_tokens)

            # 将 prompts 插入到 token 序列中（插入在 cls token 之后）
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # 遍历 block，保存前三层输出
            for blk_num, blk in enumerate(self.dino_model.blocks):
                if blk_num < 1:   # 前3层
                    # compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # 去掉 CLS 和 prompts，保留 patch tokens
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式

            return x.permute(0, 3, 1, 2)
        else:
            x = self.dino_model.forward_features(x)
            x = x['x_norm_patchtokens']  # 取无cls的输出
            bs, f, c = x.shape
            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式
            return x.permute(0, 3, 1, 2)

class DinoV2_prompt_tta(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, style_aligner, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.style_aligner = style_aligner
        self.model_name = model_name.lower()[:-4]  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('请确认layer的正确性！vitl14最高block层为23层')
        #         exit()
        elif 'dinov2'in self.model_name and 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'dinov3'in self.model_name and 'vitb16' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3_wt/dinov3_vitb16_pretrain_lvd1689m.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home/liuyh/GEO/code/CVCities-main/dinov2/dinov2_vits14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        # ---------------- visual style prompts ----------------
        embed_dim = self.dino_model.embed_dim
        self.num_prompts = 4
        self.prompt_gen = MLP_PromptGenerator(
            embed_dim=embed_dim,
            num_prompts=self.num_prompts,
        )


    def forward(self, x, style_x=None, masks=None):

        # x = self.dino_model.forward_features(x)

        # if isinstance(x, list):
        #     return self.forward_features_list(x, masks)
        
        if self.training:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)
            # x = x[0] # for DINOv3

            # 注意：保持原来代码的分片方式（cls 在 0），原作者用 x[:,2:,:] 作为 patch tokens
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C]  （与原代码一致）
            # 由 patch tokens 生成 prompts
            prompts = self.prompt_gen(patch_tokens)

            # 将 prompts 插入到 token 序列中（插入在 cls token 之后）
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # 遍历 block，保存前三层输出
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # 前1层
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # 去掉 CLS 和 prompts，保留 patch tokens
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式

            return x.permute(0, 3, 1, 2)
            # return x
        else:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)
            # x = x[0]

            # 注意：保持原来代码的分片方式（cls 在 0），原作者用 x[:,2:,:] 作为 patch tokens
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C]  （与原代码一致）
            # 由 patch tokens 生成 prompts
            prompts = self.prompt_gen(patch_tokens)

            # 将 prompts 插入到 token 序列中（插入在 cls token 之后）
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # 遍历 block，保存前三层输出
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # 前3层
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # 去掉 CLS 和 prompts，保留 patch tokens
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式

            return x.permute(0, 3, 1, 2)
            # return x
        
class DinoV2_prompt_tta_vis_style(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, style_aligner, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.style_aligner = style_aligner
        self.model_name = model_name.lower()[:-4]  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('请确认layer的正确性！vitl14最高block层为23层')
        #         exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'/home/liuyh/GEO/code/CVCities-main/dinov2/dinov2_vits14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        # ---------------- visual style prompts ----------------
        embed_dim = self.dino_model.embed_dim
        self.num_prompts = 4
        self.prompt_gen = MLP_PromptGenerator(
            embed_dim=embed_dim,
            num_prompts=self.num_prompts,
        )


    def forward(self, x, style_x=None, masks=None):
        
        if self.training:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)

            # 注意：保持原来代码的分片方式（cls 在 0），原作者用 x[:,2:,:] 作为 patch tokens
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C]  （与原代码一致）
            # 由 patch tokens 生成 prompts
            prompts = self.prompt_gen(patch_tokens)

            # 将 prompts 插入到 token 序列中（插入在 cls token 之后）
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # 遍历 block，保存前三层输出
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # 前3层
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # 去掉 CLS 和 prompts，保留 patch tokens
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式

            return x.permute(0, 3, 1, 2)
        else:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)

            # 注意：保持原来代码的分片方式（cls 在 0），原作者用 x[:,2:,:] 作为 patch tokens
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C]  （与原代码一致）

            # 由 patch tokens 生成 prompts
            prompts = self.prompt_gen(patch_tokens)

            # 将 prompts 插入到 token 序列中（插入在 cls token 之后）
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # 遍历 block，保存前三层输出
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                # if blk_num < 1:  # 前3层（blk_num == 0, 1, 2 → 所以前两层之后就 break？注意：这里只跑了 0 和 1，共两层）
                #     compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                #     x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # 去掉 CLS 和 prompts，保留 patch tokens
            prompt_tokens = x[:, 1:self.num_prompts+1, :]
            patch_tokens = x[:, 1 + self.num_prompts:, :]  # [B, F, C]

            # === 新增：计算均值和方差，拼成 B x 2C ===
            mean_prompt = prompt_tokens.mean(dim=-1)      # [B, C]
            var_prompt = prompt_tokens.var(dim=-1)        # [B, C]
            prompt_style = torch.cat([mean_prompt, var_prompt], dim=1)  # [B, 2*C]

            mean_patch = patch_tokens.mean(dim=-1)      # [B, C]
            var_patch = patch_tokens.var(dim=-1)        # [B, C]
            patch_style = torch.cat([mean_patch, var_patch], dim=1)  # [B, 2*C]

            return patch_style, prompt_style

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).to('cuda')
    model = DinoV2_prompt(model_name='dinov2_vitb14', layer1=11, facet1="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)
    # torch.onnx.export(model.dino_model, torch.randn(1, 3, 224, 224), 'dinov2_vitl14.onnx', do_constant_folding=True, verbose=False)

    print(model)
    # print(model.dino_model.cls_token)
    # print(model.dino_model.pos_embed)
    # print(model.dino_model.mask_token)
    for name, param in model.dino_model.named_parameters():
        if param.requires_grad:
            print(f'***{name}**')

    print('-' * 70)
    summary(model, (3, 224, 224), 1, 'cuda')
    print('-' * 70)

    r = model(x)

    print_nb_params(model)

    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()


