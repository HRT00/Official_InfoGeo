import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip


CUSTOM_TEMPLATES = {
    'University': 'point cloud of a big {}.'
}

    
def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, '/disk1/zhanghy/CVGL_proj/geolink/geolink/clip_ckpt')
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    h_resolution = int((cfg.new_hight-16)//cfg.vision_stride[0] + 1)
    w_resolution = int((cfg.new_width-16)//cfg.vision_stride[1] + 1)
    model = clip.build_model(state_dict or model.state_dict(), h_resolution=h_resolution, w_resolution=w_resolution, vision_stride_size=cfg.vision_stride)

    return model

class Textual_Encoder(nn.Module):

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self, prompts):
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts)
        return text_feat

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # 残差连接
    

class CLIP_Model(nn.Module):

    def __init__(self, args, clip_model):
        super().__init__()
        
        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.textual_encoder = Textual_Encoder(args, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    
    def forward_img(self, images): 
        # Image features
        image_feat = self.visual_encoder(images.to(self.dtype))
        image_feat = image_feat[-1]
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        return image_feat[:, 0, :]


    def forward_text(self, texts): 

        # Text features
        text_feat = self.textual_encoder(texts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat[torch.arange(text_feat.shape[0]), texts.argmax(dim=-1)]
        return text_feat
