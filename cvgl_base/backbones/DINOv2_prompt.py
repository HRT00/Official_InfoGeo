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
    - Input patch tokens [B, F, C].
    - Output prompts [B, N, C].
    """
    def __init__(self, embed_dim, num_prompts=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_prompts = num_prompts
        self.query_tokens = nn.Parameter(torch.randn(num_prompts, embed_dim))  # Learnable queries

        # gating mechanism
        self.gate_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward network with a small MLP
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

        # Simplified gated cross-attention
        # score = (Q * Gate(Q)) @ P^T
        q_proj = self.gate_proj(self.norm1(queries))   # [B, N, C]
        attn_logits = torch.einsum('bnc,bfc->bnf', q_proj, patch_tokens) / (C ** 0.5)  # [B, N, F]
        attn = torch.softmax(attn_logits, dim=-1)

        # Aggregate patch tokens.
        attn_output = torch.matmul(attn, patch_tokens)  # [B, N, C]

        # Residual connection and feed-forward network
        x = queries + attn_output
        x = x + self.ffn(self.norm2(x))
        return x  # [B, N, C]

class MLP_PromptGenerator(nn.Module):
    """
    Generate prompts through a nonlinear MLP mapping.
    Inspired by visual prompting in LLaVA-Next and Qwen-VL-2.
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2  # Expand the hidden dimension by a factor of two.

        # Map tokens to prompts.
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # SwiGLU is another activation option.
        #     nn.Linear(hidden_dim, num_prompts * embed_dim)
        # )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            # nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),   # SwiGLU is another activation option.
            nn.Linear(1024, num_prompts)
        )
        # self.mlp = nn.Sequential(
        #     # nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # SwiGLU is another activation option.
        #     nn.Linear(787, num_prompts)
        # )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        Output prompts [B, num_prompts, C].
        """
        B, F, C = patch_tokens.shape

        # Step 1. Average all patch tokens before projection.
        patch_tokens = self.layer_norm(patch_tokens)
        pooled = self.mlp(patch_tokens.permute(0, 2, 1))  # [B, C]

        # Step 2. Apply a nonlinear MLP projection.
        # prompts = self.mlp(pooled)  # [B, num_prompts*C]

        # step3: reshape
        # prompts = prompts.view(B, self.num_prompts, C)
        prompts = pooled.permute(0, 2, 1)

        return prompts


class MLP_PromptGenerator_(nn.Module):
    """
    Generate prompts through a nonlinear MLP mapping.
    Inspired by visual prompting in LLaVA-Next and Qwen-VL-2.
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2  # Expand the hidden dimension by a factor of two.

        # Map tokens to prompts.
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.GELU(),   # SwiGLU is another activation option.
        #     nn.Linear(hidden_dim, num_prompts * embed_dim)
        # )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(1024, num_prompts)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        Output prompts [B, num_prompts, C].
        """
        B, F, C = patch_tokens.shape

        # Step 1. Average all patch tokens before projection.
        patch_tokens = self.layer_norm(patch_tokens)
        pooled = self.mlp(patch_tokens.permute(0, 2, 1))  # [B, C]

        # Step 2. Apply a nonlinear MLP projection.
        # prompts = self.mlp(pooled)  # [B, num_prompts*C]

        # step3: reshape
        # prompts = prompts.view(B, self.num_prompts, C)
        prompts = pooled.permute(0, 2, 1)

        return prompts


class MLP_TopKPromptGenerator(nn.Module):
    """
    Top-k Prompt Generator:
    - Select up to num_prompts patch tokens using learned scores.
    - Map each selected token to a prompt embedding with an MLP.
    - If num_patches < num_prompts, pad with the mean of the selected prompts.
    Output prompts [B, num_prompts, C].
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2

        # Scalar token importance scores
        self.score_proj = nn.Linear(embed_dim, 1)

        # Token layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Map each selected token to a prompt along the last dimension.
        # Linear operates on the last dimension of an input shaped [B, k, C].
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

        # 2. Score tokens and select min(num_prompts, num_patches) tokens.
        scores = self.score_proj(tokens).squeeze(-1)  # [B, num_patches]
        k = min(self.num_prompts, num_patches)
        _, topk_idx = torch.topk(scores, k, dim=1)  # [B, k]

        # 3. Gather the top-k tokens by index.
        # Expand the indices to match the last dimension for gather.
        topk_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))  # [B, k, C]

        # 4. Map each selected token to its own prompt embedding.
        prompts_k = self.token_to_prompt(topk_tokens)  # [B, k, C]

        # 5. If k < num_prompts, fill the remaining slots with the mean prompt.
        if k < self.num_prompts:
            # Pad with the mean of the selected prompts.
            filler = prompts_k.mean(dim=1, keepdim=True).expand(B, self.num_prompts - k, C)  # [B, num_prompts-k, C]
            prompts = torch.cat([prompts_k, filler], dim=1)  # [B, num_prompts, C]
        else:
            prompts = prompts_k  # The prompt count already equals num_prompts.

        return prompts  # [B, num_prompts, C]


class ChannelPromptGenerator(nn.Module):
    """
    Channel attention followed by MLP prompt mapping.
    Compute channel weights (B, 1, C), then map (B, F, C) to (B, num_prompts, C).
    """
    def __init__(self, embed_dim, num_prompts=8, hidden_dim=None):
        super().__init__()
        self.num_prompts = num_prompts
        hidden_dim = hidden_dim or embed_dim * 2

        # Channel attention maps (B, F, C) to (B, 1, C).
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        # Prompt generation MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Fusion layer maps F to num_prompts.
        self.fusion = nn.Linear(1, num_prompts)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, F, C]
        Output prompts [B, num_prompts, C].
        """
        B, F, C = patch_tokens.shape

        # Step 1. Generate channel attention from (B, F, C) to (B, 1, C).
        # Generate an attention weight for each channel.
        channel_weights = self.channel_attn(patch_tokens.mean(dim=1, keepdim=True))  # [B, 1, C]

        # Step 2. Apply channel attention.
        pooled = channel_weights * patch_tokens  # [B, F, C]

        # Step 3. Apply the MLP to each token.
        pooled = self.mlp(pooled)  # [B, F, C]

        # Step 4. Reduce F to num_prompts.
        # Average along F, then map to num_prompts.
        pooled_mean = pooled.mean(dim=2, keepdim=True)  # [B, F, 1]
        weights = self.fusion(pooled_mean.transpose(1, 2))  # [B, num_prompts, F]
        attn = F.softmax(weights, dim=-1)  # [B, num_prompts, F]

        # Step 5. Aggregate the final prompts.
        prompts = torch.bmm(attn, pooled)  # [B, num_prompts, C]

        return prompts

class SwiGLU(nn.Module):
    """
    SwiGLU activation from the PaLM paper, also used in Llama and Mistral.
    Combines a SiLU gate with a linear branch.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 2 * D]
        Output shape [..., D].
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class MLP(nn.Module):
    """
    Configurable feed-forward network (FFN).
    Based on FFN designs used in models such as ViT and Llama.
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
        
        # SwiGLU requires twice the output width in the first linear layer.
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

# PatchPromptGenerator

class PatchPromptGeneratorV2(nn.Module):
    """
    Patch-Level Attention Prompt Generator (V2 - 2025 Style)

    Aggregate image patch features into a small set of visual prompts using attention.
    Inspired by multimodal models such as LLaVA-Next, Qwen2-VL, and InternVL2.

    Features include the following.
    - Configurable MLP activation, including SwiGLU.
    - Optional residual connection after the final projection.
    - Configurable hidden dimension through mlp_ratio.
    - Type annotations and documentation.
    """
    def __init__(
        self,
        embed_dim: int,
        num_prompts: int = 8,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = SwiGLU, # Use SwiGLU by default.
        add_residual: bool = True,
    ):
        """
        Args:
            embed_dim (int): Input patch token dimension C.
            num_prompts (int): Number of visual prompts to generate.
            mlp_ratio (float): Ratio of the MLP hidden dimension to embed_dim.
            act_layer (Type[nn.Module]): Activation class used in the FFN.
            add_residual (bool): Whether to add a residual connection after projection.
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.add_residual = add_residual
        hidden_dim = int(embed_dim * mlp_ratio)

        # Patch Attention Scorer generates weights from each patch to each prompt.
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

        # Prompt Projector refines the aggregated features.
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
            patch_features (torch.Tensor): Input image patch features of shape [B, F, C].
                                           B = Batch size
                                           F = Number of patches (e.g., 256 for 224x224 image)
                                           C = Embedding dimension

        Returns:
            torch.Tensor: Generated visual prompts of shape [B, num_prompts, C].
        """

        # Step 1. Compute attention scores from each patch to each prompt.
        # [B, F, C] -> [B, F, num_prompts]
        attention_scores = self.scorer(patch_features)

        # Step 2. Normalize weights with softmax along the patch dimension F.
        # transpose: [B, num_prompts, F]
        # softmax:   [B, num_prompts, F]
        attention_weights = F.softmax(attention_scores.transpose(1, 2), dim=-1)

        # Step 3. Aggregate F patch features into num_prompts vectors with a weighted sum.
        # torch.bmm([B, num_prompts, F], [B, F, C]) -> [B, num_prompts, C]
        aggregated_prompts = torch.bmm(attention_weights, patch_features)

        # Step 4. Transform and refine the features with the projector.
        # [B, num_prompts, C] -> [B, num_prompts, C]
        refined_prompts = self.projector(aggregated_prompts)

        # Step 5. Optionally add a residual connection.
        if self.add_residual:
            prompts = aggregated_prompts + refined_prompts
        else:
            prompts = refined_prompts
            
        return prompts

class DinoV2_prompt(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        Extract features from intermediate DINOv2 layers.
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
        self.model_name = model_name.lower()[:-7]  # Convert the model name to lowercase.
        self.layer1 = layer1

        self.pretrained = pretrained  # Whether to use pretrained weights.
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('Check the layer setting. The maximum block index for vitl14 is 23.')
        #         exit()
        elif 'dinov2'in self.model_name and 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'dinov3'in self.model_name and 'vitb16' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3_wt/dinov3_vitb16_pretrain_lvd1689m.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
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

            # Preserve the token slicing, with CLS at index 0 and patches selected by x[:, 2:, :].
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C], following the original token layout.
            # Generate prompts from patch tokens.
            prompts = self.prompt_gen(patch_tokens)

            # Insert prompts after the CLS token.
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # Iterate over transformer blocks and collect intermediate outputs.
            for blk_num, blk in enumerate(self.dino_model.blocks):
                if blk_num < 1:   # Select the early blocks.
                    # compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # Remove CLS and prompt tokens, retaining patch tokens.
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.

            return x.permute(0, 3, 1, 2)
        else:
            x = self.dino_model.forward_features(x)
            x = x['x_norm_patchtokens']  # Select patch outputs without the CLS token.
            bs, f, c = x.shape
            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.
            return x.permute(0, 3, 1, 2)

class DinoV2_prompt_tta(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        Extract features from intermediate DINOv2 layers.
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
        self.model_name = model_name.lower()[:-4]  # Convert the model name to lowercase.
        self.layer1 = layer1

        self.pretrained = pretrained  # Whether to use pretrained weights.
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('Check the layer setting. The maximum block index for vitl14 is 23.')
        #         exit()
        elif 'dinov2'in self.model_name and 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'dinov3'in self.model_name and 'vitb16' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov3_wt/dinov3_vitb16_pretrain_lvd1689m.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
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

            # Preserve the token slicing, with CLS at index 0 and patches selected by x[:, 2:, :].
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C], following the original token layout.
            # Generate prompts from patch tokens.
            prompts = self.prompt_gen(patch_tokens)

            # Insert prompts after the CLS token.
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # Iterate over transformer blocks and collect intermediate outputs.
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # Select the first block.
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # Remove CLS and prompt tokens, retaining patch tokens.
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.

            return x.permute(0, 3, 1, 2)
            # return x
        else:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)
            # x = x[0]

            # Preserve the token slicing, with CLS at index 0 and patches selected by x[:, 2:, :].
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C], following the original token layout.
            # Generate prompts from patch tokens.
            prompts = self.prompt_gen(patch_tokens)

            # Insert prompts after the CLS token.
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # Iterate over transformer blocks and collect intermediate outputs.
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # Select the early blocks.
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # Remove CLS and prompt tokens, retaining patch tokens.
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.

            return x.permute(0, 3, 1, 2)
            # return x
        
class DinoV2_prompt_tta_vis_style(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        Extract features from intermediate DINOv2 layers.
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
        self.model_name = model_name.lower()[:-4]  # Convert the model name to lowercase.
        self.layer1 = layer1

        self.pretrained = pretrained  # Whether to use pretrained weights.
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2_wt/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        # elif 'vitl14_reg' in self.model_name:
        #     self.dino_model = torch.hub.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
        #     self.dino_model.load_state_dict(torch.load(r'/home2/zhanghy/CVCities-main-pointclip/dinov2/dinov2_vitl14_reg4_pretrain.pth'))
        #     if self.layer1 > 23:
        #         print('Check the layer setting. The maximum block index for vitl14 is 23.')
        #         exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
            self.dino_model.load_state_dict(torch.load(r'/disk1/zhanghy/CVGL_proj/geolink/geolink/dinov2_wt/dinov2_vitb14_pretrain.pth'), strict=False)
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'/root/group-trainee/zhy/train/code/dinov2-main', self.model_name, trust_repo=True, source='local', pretrained=False)  # Load the DINOv2 backbone.
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

            # Preserve the token slicing, with CLS at index 0 and patches selected by x[:, 2:, :].
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C], following the original token layout.
            # Generate prompts from patch tokens.
            prompts = self.prompt_gen(patch_tokens)

            # Insert prompts after the CLS token.
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # Iterate over transformer blocks and collect intermediate outputs.
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                if blk_num < 1:   # Select the early blocks.
                    compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :], style_x[blk_num])
                    x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # Remove CLS and prompt tokens, retaining patch tokens.
            x = x[:, 1 + self.num_prompts:, :]

            bs, f, c = x.shape

            x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.

            return x.permute(0, 3, 1, 2)
        else:
            x = self.dino_model.prepare_tokens_with_masks(x, masks)

            # Preserve the token slicing, with CLS at index 0 and patches selected by x[:, 2:, :].
            cls_token = x[:, :1, :]        # [B,1,C]
            patch_tokens = x[:, 2:, :]     # [B, F, C], following the original token layout.

            # Generate prompts from patch tokens.
            prompts = self.prompt_gen(patch_tokens)

            # Insert prompts after the CLS token.
            x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

            # Iterate over transformer blocks and collect intermediate outputs.
            for blk_num, blk in enumerate(self.dino_model.blocks):
                x = blk(x)
                # if blk_num < 1:  # Select early blocks before the loop terminates.
                #     compx_aligned = self.style_aligner(blk_num, x[:, 1:self.num_prompts+1, :])
                #     x = torch.cat([x[:, :1, :], compx_aligned, x[:, self.num_prompts+1:, :]], dim=1)

            x = self.dino_model.norm(x)

            # Remove CLS and prompt tokens, retaining patch tokens.
            prompt_tokens = x[:, 1:self.num_prompts+1, :]
            patch_tokens = x[:, 1 + self.num_prompts:, :]  # [B, F, C]

            # Concatenate the mean and variance into a B x 2C descriptor.
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


