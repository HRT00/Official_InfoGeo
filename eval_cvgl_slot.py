import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from cvgl_base.dataset import gta
from cvgl_base.dataset.sues import SUESDatasetEval
from cvgl_base.dataset.university import get_transforms
from cvgl_base.evaluate.gta import evaluate as evaluate_gta
from cvgl_base.evaluate.university import evaluate
from cvgl_base.model_slot_dias import TimmModel_slot
from cvgl_base.utils import setup_system


@dataclass
class Configuration:
    # Model
    model: str = 'dinov2_vitb14_MixVPR'
    backbone_arch: str = 'dinov2_vitb14'
    layer1: int = 7
    vfm_dim: int = 768
    emb_dim: int = 1024
    num_slots: int = 16
    iters: int = 3
    alpha: float = 0.8
    img_size: int = 448
    view_token: Optional[bool] = None

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    normalize_features: bool = True
    eval_gallery_n: int = -1
    target: str = 'sues'
    direction: str = 'D2S'
    test_root: Path = Path('data/SUES-200/SUES-200-512x512/Testing/150')
    checkpoint_start: Optional[Path] = None
    num_workers: int = 0 if os.name == 'nt' else 7
    seed: int = 1
    device: str = 'cuda'
    gpu_ids: tuple = (0,)


def add_evaluation_arguments(parser):
    parser.add_argument('--target', choices=('sues', 'gta'))
    parser.add_argument('--direction', choices=('D2S', 'S2D'))
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--checkpoint', dest='checkpoint_start', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--num-slots', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--view-token', action=argparse.BooleanOptionalAction)


def build_model(config, pretrained):
    state_dict = None
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        state_dict = torch.load(config.checkpoint_start, map_location='cpu', weights_only=True)
        token_keys = ('model.backbone.dino_model.view_token', 'model.backbone.dino_model.pos_view_embed')
        present = [key in state_dict for key in token_keys]
        if any(present) != all(present):
            raise ValueError("The checkpoint contains an incomplete view token.")
        if config.view_token is None:
            config.view_token = all(present)
        elif not config.view_token and all(present):
            for key in token_keys:
                del state_dict[key]
    elif config.view_token is None:
        config.view_token = True

    print("\nModel:", config.model)
    model = TimmModel_slot(model_name=config.model, pretrained=pretrained,
                           backbone_arch=config.backbone_arch, vfm_dim=config.vfm_dim,
                           emb_dim=config.emb_dim, img_size=config.img_size, layer1=config.layer1,
                           num_slots=config.num_slots, iters=config.iters, alpha=config.alpha,
                           use_view_token=config.view_token)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    print("DINOv2 view token:", config.view_token)
    return model.to(config.device)


def evaluation_data(config, data_config, batch_size):
    img_size = (config.img_size, config.img_size)
    if config.target == 'sues':
        val_transforms, _, _ = get_transforms(img_size, **data_config)
        query_view, gallery_view = ('drone', 'satellite') if config.direction == 'D2S' else ('satellite', 'drone')
        query_dataset_test = SUESDatasetEval(data_folder=config.test_root / f'query_{query_view}',
                                             mode='query', transforms=val_transforms)
        gallery_dataset_test = SUESDatasetEval(data_folder=config.test_root / f'gallery_{gallery_view}',
                                               mode='gallery', transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n)
        metric = evaluate
        options = dict(ranks=[1, 5, 10], step_size=1000, cleanup=True)
    else:
        if config.direction != 'D2S':
            raise ValueError("The GTA cross-area protocol requires D2S.")
        val_transforms, _, _ = gta.get_transforms(img_size, sat_rot=True, **data_config)
        dataset_args = dict(data_root=config.test_root.resolve(), pairs_meta_file='cross-area-drone2sate-test.json',
                            transforms=val_transforms, mode='pos', sate_img_dir='satellite', query_mode='D2S')
        query_dataset_test = gta.GTADatasetEval(view='drone', **dataset_args)
        gallery_dataset_test = gta.GTADatasetEval(view='sate', **dataset_args)
        metric = evaluate_gta
        options = dict(query_list=query_dataset_test.images_name, gallery_list=gallery_dataset_test.images_name,
                       query_center_loc_xy_list=query_dataset_test.images_center_loc_xy,
                       gallery_center_loc_xy_list=gallery_dataset_test.images_center_loc_xy,
                       pairs_dict=query_dataset_test.pairs_drone2sate_dict, dis_threshold_list=list(range(10, 501, 10)))

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    if not len(query_dataset_test) or not len(gallery_dataset_test):
        raise ValueError(f"Empty query or gallery at {config.test_root}")
    loader_args = dict(batch_size=batch_size, num_workers=config.num_workers, shuffle=False, pin_memory=True)
    return (metric, DataLoader(query_dataset_test, **loader_args),
            DataLoader(gallery_dataset_test, **loader_args), options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained InfoGeo checkpoint.")
    add_evaluation_arguments(parser)
    parser.add_argument('--eval-batch-size', '--batch-size', dest='batch_size', type=int)
    config = parser.parse_args(namespace=Configuration())
    if config.checkpoint_start is None:
        parser.error('Set checkpoint_start in Configuration or supply --checkpoint.')

    # Model
    setup_system(seed=config.seed, cudnn_benchmark=True, cudnn_deterministic=False)
    model = build_model(config, pretrained=False)

    # DataLoader
    metric, query_dataloader_test, gallery_dataloader_test, options = evaluation_data(
        config, model.get_config(), config.batch_size)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Evaluate
    r1_test = metric(config=config, model=model, query_loader=query_dataloader_test,
                     gallery_loader=gallery_dataloader_test, **options)
    if not math.isfinite(float(r1_test)):
        raise ValueError("Checkpoint evaluation did not produce a finite retrieval score.")
