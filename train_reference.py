import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from cvgl_base.dataset import gta
from cvgl_base.dataset.university import U1652DatasetTrain2D, get_transforms
from cvgl_base.loss.loss import InfoNCE
from cvgl_base.trainer_slot import train
from cvgl_base.utils import Logger, setup_system
from eval_cvgl_slot import Configuration as EvaluationConfiguration
from eval_cvgl_slot import add_evaluation_arguments, build_model, evaluation_data


@dataclass
class Configuration(EvaluationConfiguration):
    # Training
    epochs: int = 1
    batch_size: int = 4
    batch_size_eval: int = 128
    first_eval: int = 1
    eval_every_n_epoch: int = 1
    custom_sampling: bool = True

    # Loss and optimizer
    rec: float = 0.05
    dis: float = 0.85
    reg: float = 0.05
    cacs_weight: float = 1.0
    stable_graph_gradients: bool = False
    label_smoothing: float = 0.1
    clip_grad: float = 100.
    lr: float = 0.00065
    backbone_lr_scale: float = 1.0
    scheduler: str = 'cosine'
    warmup_epochs: float = 0.15

    # Dataset
    source: str = 'dense'
    data_folder: Path = Path('data/DenseUAV')
    prob_flip: float = 0.5
    dense_drone_height: Optional[int] = None
    square_crop_dense_drone: bool = False
    university_drone_view_range: Optional[Tuple[int, int]] = None
    model_path: Path = Path('logs/reference_retrain')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train InfoGeo and evaluate after each scheduled epoch.")
    add_evaluation_arguments(parser)
    parser.add_argument('--source', choices=('dense', 'university', 'gta'))
    parser.add_argument('--train-root', dest='data_folder', type=Path)
    parser.add_argument('--output', dest='model_path', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--first-eval', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--eval-batch-size', dest='batch_size_eval', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--backbone-lr-scale', type=float)
    parser.add_argument('--warmup-epochs', type=float)
    parser.add_argument('--rec', type=float)
    parser.add_argument('--dis', type=float)
    parser.add_argument('--reg', type=float)
    parser.add_argument('--cacs-weight', type=float)
    parser.add_argument('--stable-graph-gradients', action='store_true')
    parser.add_argument('--dense-drone-height', type=int, choices=(80, 90, 100))
    parser.add_argument('--square-crop-dense-drone', action='store_true')
    parser.add_argument('--university-drone-view-range', type=int, nargs=2, metavar=('FIRST', 'LAST'))
    config = parser.parse_args(namespace=Configuration())
    if len(config.gpu_ids) != 1:
        parser.error("Slot training uses one GPU. Select it with CUDA_VISIBLE_DEVICES.")
    if (config.dense_drone_height or config.square_crop_dense_drone) and config.source != 'dense':
        parser.error('DenseUAV height and cropping require --source dense.')
    if config.university_drone_view_range is not None:
        first, last = config.university_drone_view_range
        if config.source != 'university' or not 1 <= first <= last <= 54:
            parser.error('University view range requires --source university and 1 <= FIRST <= LAST <= 54.')
    if not math.isfinite(config.backbone_lr_scale) or config.backbone_lr_scale < 0:
        parser.error('--backbone-lr-scale must be finite and nonnegative.')

    model_path = config.model_path / config.model / datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')
    model_path.mkdir(parents=True)
    sys.stdout = Logger(str(model_path / 'log.txt'))
    setup_system(seed=config.seed, cudnn_benchmark=True, cudnn_deterministic=False)

    # Model
    model = build_model(config, pretrained=True)
    data_config = model.get_config()
    (model_path / 'config.json').write_text(json.dumps(asdict(config), default=str, indent=2))

    # DataLoader
    img_size = (config.img_size, config.img_size)
    if config.source == 'gta':
        _, train_sat_transforms, train_drone_transforms = gta.get_transforms(img_size, sat_rot=True, **data_config)
        train_dataset = gta.GTADatasetTrain(
            'cross-area-drone2sate-train.json', config.data_folder, transforms_query=train_drone_transforms,
            transforms_gallery=train_sat_transforms, mode='pos', group_len=2, shuffle_batch_size=config.batch_size)
    else:
        _, train_sat_transforms, train_drone_transforms = get_transforms(img_size, **data_config)
        # The DenseUAV reference applies the satellite transform to the drone query.
        query_transform, gallery_transform = train_sat_transforms, train_drone_transforms
        if config.source == 'university':
            query_transform, gallery_transform = train_drone_transforms, train_sat_transforms
        train_dataset = U1652DatasetTrain2D(
            query_folder=config.data_folder / 'train/drone', gallery_folder=config.data_folder / 'train/satellite',
            transforms_query=query_transform, transforms_gallery=gallery_transform, prob_flip=config.prob_flip,
            shuffle_batch_size=config.batch_size,
            fixed_drone_filename=f'H{config.dense_drone_height}.JPG' if config.dense_drone_height else None,
            square_crop_query=config.square_crop_dense_drone, drone_view_range=config.university_drone_view_range,
            rotate_drone_views=config.university_drone_view_range is not None,
            drone_view_seed=config.seed if config.university_drone_view_range is not None else None)
    if not len(train_dataset):
        raise ValueError(f"Empty training dataset at {config.data_folder}")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling, pin_memory=True)
    evaluate, query_dataloader_test, gallery_dataloader_test, eval_options = evaluation_data(
        config, data_config, config.batch_size_eval)

    # Loss
    loss_function = {'InfoNCE': InfoNCE(torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing),
                                        device=config.device)}

    # Optimizer
    parameters = model.parameters()
    if config.backbone_lr_scale != 1.0:
        backbone = model.model.backbone
        backbone_weights = set(backbone.parameters())
        parameters = [
            {'params': backbone.parameters(), 'lr': config.lr * config.backbone_lr_scale},
            {'params': [p for p in model.parameters() if p not in backbone_weights]},
        ]
    optimizer = torch.optim.SGD(parameters, lr=config.lr)
    scaler = GradScaler(init_scale=1024)

    # Scheduler uses the number of batches before the first custom shuffle.
    steps_per_epoch = len(train_dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, steps_per_epoch * config.warmup_epochs,
                                                steps_per_epoch * config.epochs)
    print(f"Training pairs: {len(train_dataset)}, scheduled steps: {steps_per_epoch * config.epochs}")

    # Train
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()
    best_score = 0.
    for epoch in range(1, config.epochs + 1):
        train_loss = train(config, model, dataloader=train_dataloader, loss_function=loss_function,
                           optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        print("Epoch: {}, Train Loss = {:.3f},  Lr = {:.6f}".format(
            epoch, train_loss, optimizer.param_groups[0]['lr']))

        if (epoch >= config.first_eval and epoch % config.eval_every_n_epoch == 0) or epoch == config.epochs:
            r1_test = evaluate(config=config, model=model, query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, **eval_options)
            if r1_test >= best_score:
                best_score = r1_test
                torch.save(model.state_dict(), model_path / f'weights_e{epoch}_{r1_test:.4f}.pth')
        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    torch.save(model.state_dict(), model_path / 'weights_end.pth')
