import os

import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup


from cvgl_base.dataset.university import get_transforms
# from cvgl_base.dataset.denseuav import DenseUAVDatasetEval
from cvgl_base.dataset.gta import GTADatasetTrain, GTADatasetEval, get_transforms
from cvgl_base.utils import setup_system, Logger
from cvgl_base.trainer_slot import train
from cvgl_base.evaluate.gta import evaluate
from cvgl_base.loss.loss import InfoNCE, WeightedInfoNCE
from cvgl_base.loss.blocks_infoNCE import blocks_InfoNCE
from cvgl_base.loss.DSA_loss import DSA_loss
from cvgl_base.loss.supcontrast import SupConLoss
from cvgl_base.model_slot_dias import TimmModel_slot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
# 忽略特定的 libpng 警告
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")

@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    pretrained = True
    layer1 = 7
    use_cls = True
    norm_descs = True

    # slot attention
    vfm_dim = 768
    emb_dim = 1024
    num_slots = 16
    iters = 3

    # fusion
    alpha = 0.8

    # Override model image size
    img_size: int = 448
    new_hight = 448
    new_width = 448

    # Training
    mixed_precision: bool = True
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 7
    batch_size: int = 16  # keep in mind real_batch_size = 2 * batch_size    # 8 for vitb14 | 2 for vitg14 
    verbose: bool = True
    gpu_ids: tuple = (0,)  # GPU ids for training

    train_in_group: bool = True
    group_len = 2

    # Eval
    batch_size_eval: int = 128   # 64 for vitb14 | 16 for vitg14 | 32 for vitl14
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing
    use_sgd = True

    # Loss
    rec = 0.05
    dis = 0.85
    reg = 0.05
    label_smoothing: float = 0.1
    k: float = 5

    # Learning Rate
    lr: float = 0.000685  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.25
    lr_end: float = 0.00010  # only for "polynomial"

    # Dataset
    data_root: str = r"Your_Path"
    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'

    query_mode: str = "D2S"               # Retrieval in Drone to Satellite
    train_mode: str = "pos"       # Train with positive + semi-positive pairs
    test_mode: str = "pos"                # Test with positive pairs

    # Augment Images
    prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "./logs/slot_gta_cross"

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 7

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False

# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%Y-%m-%d_%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    print("\nModel: {}".format(config.model))

    model = TimmModel_slot(model_name=config.model,
                      pretrained=True, backbone_arch=config.backbone_arch, vfm_dim=config.vfm_dim, emb_dim=config.emb_dim,
                      img_size=config.img_size, layer1=config.layer1, num_slots=config.num_slots, iters=config.iters, alpha=config.alpha)
    print(model)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]

    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    if 'cross-area' in config.train_pairs_meta_file:
        sat_rot = True
    else:
        sat_rot = False
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms(img_size, mean=mean, std=std, sat_rot=sat_rot)
                                                                                                                
    # Train
    train_dataset = GTADatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms,
                                    group_len=config.group_len,
                                    prob_flip=config.prob_flip,
                                    shuffle_batch_size=config.batch_size,
                                    mode=config.train_mode,
                                    train_ratio=config.train_ratio,
                                    )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    # Test query
    if config.query_mode == 'D2S':
        query_view = 'drone'
        gallery_view = 'sate'
    else:
        query_view = 'sate'
        gallery_view = 'drone'

    query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view=query_view,
                                        transforms=val_transforms,
                                        mode=config.test_mode,
                                        sate_img_dir=config.sate_img_dir,
                                        query_mode=config.query_mode,
                                        )
    if config.query_mode == 'D2S':                                    
        pairs_dict = query_dataset_test.pairs_drone2sate_dict
    else:
        pairs_dict = gallery_dataset_test.pairs_sate2drone_dict
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                          pairs_meta_file=config.test_pairs_meta_file,
                                          view=gallery_view,
                                          transforms=val_transforms,
                                          mode=config.test_mode,
                                          sate_img_dir=config.sate_img_dir,
                                          query_mode=config.query_mode,
                                         )
    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function1 = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )
    # loss_function1 = WeightedInfoNCE(
    #     loss_function=loss_fn,
    #     device=config.device,
    #     label_smoothing=config.label_smoothing,
    #     k=config.k,
    # )
    loss_function2 = blocks_InfoNCE(loss_function=loss_fn, device=config.device,)
    loss_function3 = DSA_loss(loss_function=loss_fn, device=config.device,)
    loss_function4 = SupConLoss(device=config.device)
    
    loss_function = {
        'InfoNCE': loss_function1,
        'blocks_InfoNCE': loss_function2,
        'DSA': loss_function3,
        'SupCon': loss_function4,
    }

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#        
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    for epoch in range(1, config.epochs + 1):

        print("\n{}[{}/Epoch: {}]{}".format(30*"-",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),  epoch, 30*"-"))

        train_loss = train(config,
                           model,
                           loss_function=loss_function,
                           dataloader=train_dataloader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f},  Lr = {:.6f}".format(epoch, 
                                                                    train_loss,
                                                                    optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch ==0 and epoch >= 2) or epoch == config.epochs:

            # For Test Log (distance threshold) 
            dis_threshold_list = None
            if 'cross' in config.test_pairs_meta_file:
                ####### Cross-area for total 500m/10m
                print("cross-area eval")
                dis_threshold_list = [10*(i+1) for i in range(50)]
            else:
                ####### Same-area for total 200m/4m
                print("same-area eval")
                dis_threshold_list = [4*(i+1) for i in range(50)]
            
            print("\n{}[{}]{}".format(30*"-", "Evaluating GTA-UAV", 30*"-"))

            r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           query_list=query_img_list,
                           gallery_list=gallery_img_list,
                           pairs_dict=pairs_dict,
                           ranks_list=[1, 5, 10],
                           query_center_loc_xy_list=query_center_loc_xy_list,
                           gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                           dis_threshold_list=dis_threshold_list,
                           cleanup=True,
                           plot_acc_threshold=False,
                           top10_log=False)

            if r1_test >= best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                                '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))


        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
        
