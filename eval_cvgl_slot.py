import os

import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from cvgl_base.dataset.sues import get_transforms
from cvgl_base.dataset.sues import SUESDatasetEval
from cvgl_base.evaluate.sues import evaluate
from cvgl_base.model_slot_dias import TimmModel_slot


@dataclass
class Configuration:
    # Model
    model: str = 'dinov2_vitb14_MixVPR'
    layer1 = 7
    backbone_arch = 'dinov2_vitb14'

    # slot attention
    vfm_dim = 768
    emb_dim = 1024
    num_slots = 16
    iters = 3

    # fusion
    alpha = 0.8
    
    # Override model image size
    img_size: int = 448
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'SUES-D2S'           # Two Directions
    data_folder: str = "Your_path"
    
    # Checkpoint to start from
    checkpoint_start = 'Your_CKPT'   # Using University or DenseUAV Weights for Cross-Dataset Evaluation
    num_workers: int = 0 if os.name == 'nt' else 7
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'SUES-D2S':
    config.query_folder_test = f'{config.data_folder}/query_drone'
    config.gallery_folder_test = f'{config.data_folder}/gallery_satellite'
elif config.dataset == 'SUES-S2D': 
    config.query_folder_test = f'{config.data_folder}/query_satellite' 
    config.gallery_folder_test = f'{config.data_folder}/gallery_drone'
else:
    raise ValueError(f"Unknow Scenarios: {config.dataset}")


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    model = TimmModel_slot(model_name=config.model,
                      pretrained=True, backbone_arch=config.backbone_arch, vfm_dim=config.vfm_dim, emb_dim=config.emb_dim,
                      img_size=config.img_size, layer1=config.layer1, num_slots=config.num_slots, iters=config.iters, alpha=config.alpha)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # load pretrained Checkpoint    
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


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    
    # Reference Satellite Images
    query_dataset_test = SUESDatasetEval(data_folder=config.query_folder_test,
                                           mode="query",
                                           transforms=val_transforms,
                                            )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = SUESDatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   

    print("\n{}[{}]{}".format(30*"-", "CVGL_eval", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
 
