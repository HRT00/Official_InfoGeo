# --<utf-8>--


import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitl14_reg", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_self(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        Extract features from intermediate DINOv2 layers.
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True, use_view_token=True) -> None:
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
        self.model_name = model_name.lower()  # Convert the model name to lowercase.
        self.layer1 = layer1

        self.pretrained = pretrained  # Whether to use pretrained weights.
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        root = Path(__file__).resolve().parents[2]
        source = root / "vendor" / "dinov2-main"
        if not (source / "hubconf.py").is_file():
            raise FileNotFoundError(f"Extract dinov2-main.zip into {root / 'vendor'}")
        self.dino_model = torch.hub.load(str(source), self.model_name, trust_repo=True,
                                         source="local", pretrained=False, use_view_token=use_view_token)
        if self.layer1 >= len(self.dino_model.blocks):
            raise ValueError(f"Layer {self.layer1} exceeds the depth of {self.model_name}")
        if pretrained:
            default_weights = root / "checkpoints" / f"{self.model_name}_pretrain.pth"
            weights = Path(os.environ.get("DINOV2_WEIGHTS", default_weights)).expanduser()
            if not weights.is_file():
                raise FileNotFoundError(f"Set DINOV2_WEIGHTS to the pretrained backbone file or place it at {default_weights}")
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.dino_model.load_state_dict(state, strict="vitb14" not in self.model_name)

        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()


    def forward(self, x, masks=None):

        x = self.dino_model.forward_features(x)

        # if isinstance(x, list):
        #     return self.forward_features_list(x, masks)
        #
        # x = self.dino_model.prepare_tokens_with_masks(x, masks)
        # for blk_num in range(len(self.dino_model.blocks)):
        #     x = self.dino_model.blocks[blk_num](x)
        #     if blk_num == 7:
        #         x1 = x
        #     if blk_num == 9:
        #         x2 = x
        #     if blk_num == 11:
        #         x3 = x
        # x = torch.cat((x1, x2, x3), dim=2)

        view_x = x["x_norm_viewtoken"]
        cls_x = x['x_norm_clstoken']
        x = x['x_norm_patchtokens']  # Select patch outputs without the CLS token.
        bs, f, c = x.shape

        # x = (x + x.mean(dim=2, keepdim=True)) * 0.5

        x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # Reshape the token sequence into a spatial feature map.
        return x.permute(0, 3, 1, 2), view_x, cls_x



def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).to('cuda')
    model = DinoV2_self(model_name='dinov2_vitb14', layer1=11, facet1="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)
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


