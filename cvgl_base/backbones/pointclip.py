import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from .mv_utils_fs import PCViews


CUSTOM_TEMPLATES = {
    'University': 'point cloud of a big {}.'
}

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)
    
def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

class Textual_Encoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)
        return text_feat

class PointCLIP_Model(nn.Module):

    def __init__(self, args, clip_model):
        super().__init__()
        
        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        # self.textual_encoder = Textual_Encoder(args, classnames, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Multi-view projection
        self.num_views = args.num_views
        pc_views = PCViews()
        self.get_img = pc_views.get_img

        
        # inter-view Adapter
        self.adapter = Adapter(args).to(clip_model.dtype)
        
        # MoEs
        # self.moe_ln = nn.LayerNorm(normalized_shape=[5120])
        
        # self.reid_moe = nn.Sequential(nn.Linear(5120, 1),
        #                               nn.ReLU())

        # Store features for post-process view-weight search
        self.store = False
        self.feat_store = []
        self.label_store = []

    
    def forward(self, pc): 
        # pc.shape: B*D*3
        # Project to multi-view depth maps
        images = self.mv_proj(pc).type(self.dtype)  # B*1024*3

        # Image features
        image_feat = self.visual_encoder(images)
        image_feat = self.adapter(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)   

        # Store for the best ckpt
        if self.store:
            self.feat_store.append(image_feat)
            # self.label_store.append(label)

        # # Text features
        # text_feat = self.textual_encoder()
        # text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # # Classification logits
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_feat @ text_feat.t() * 1.

        # return logits
        # image_feat = image_feat.reshape(-1, 3, image_feat.shape[-1])
        
        # img1_feat = self.moe_ln(image_feat[:, 0])
        # img2_feat = self.moe_ln(image_feat[:, 1])
        # img3_feat = self.moe_ln(image_feat[:, 2])
        
        # img_feats = torch.stack([img1_feat, img2_feat, img3_feat]).permute(1, 0, 2)
        # score_all = self.reid_moe(img_feats).reshape(image_feat.shape[0], 3)
        # score_all = score_all.softmax(dim=1)
        
        # moe_feat = torch.einsum('bsd,bs->bd', image_feat, score_all)  # bs*feat_dim
        
        # return moe_feat
        return image_feat

    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        return img
    
class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self, args):
        super().__init__()

        self.num_views = args.num_views
        self.in_features = args.backbone_channel
        self.adapter_ratio = args.adapter_ratio
        self.fusion_init = args.adapter_init
        self.dropout = args.adapter_dropout

        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        self.global_f = nn.Sequential(
                BatchNormPoint(self.in_features),
                nn.Dropout(self.dropout),
                nn.Flatten(),
                nn.Linear(in_features=self.in_features * self.num_views,
                          out_features=self.in_features),
                nn.BatchNorm1d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout))

        self.view_f = nn.Sequential(
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features),
                nn.ReLU(),
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features * self.num_views),
                nn.ReLU())


    def forward(self, feat):

        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        res_feat = feat.reshape(-1, self.num_views * self.in_features)
        
        # Global feature
        global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
        # View-wise adapted features
        view_feat = self.view_f(global_feat)
        
        img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)

        return img_feat