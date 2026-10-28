import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
from .loss.loss import slot_infoNCE, relational_distillation_3d_loss, reg_InfoNCE

# mse_loss = nn.MSELoss()

def cal_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
          
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, weight in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                # weight = weight.to(train_config.device)
            
                # Forward pass
                # feats1, feats2, features1, features2, slots_g, slots_s, r_g, r_s = model(query, reference)
                feats, des, attent, recon = model(query, reference)

                # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                #     loss_sim_self1 = loss_function['InfoNCE'](features1, features1, model.module.logit_scale.exp())
                #     loss_sim_self2 = loss_function['InfoNCE'](features2, features2, model.module.logit_scale.exp())
                #     loss_sim_g2s = loss_function['InfoNCE'](features1, features2, model.module.logit_scale.exp()) + loss_function['InfoNCE'](slots_g, slots_s, model.module.logit_scale.exp()) 

                # else:
                #     loss_sim_self1 = loss_function['InfoNCE'](features1, features1, model.logit_scale.exp())
                #     loss_sim_self2 = loss_function['InfoNCE'](features2, features2, model.logit_scale.exp())
                #     loss_sim_g2s = loss_function['InfoNCE'](features1, features2, model.logit_scale.exp())

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss_sim_self1 = loss_function['InfoNCE'](des[0], des[0], model.module.logit_scale.exp(), weight) + loss_function['InfoNCE'](des[2], des[2], model.module.logit_scale.exp(), weight)
                    loss_sim_self2 = loss_function['InfoNCE'](des[1], des[1], model.module.logit_scale.exp(), weight) + loss_function['InfoNCE'](des[3], des[3], model.module.logit_scale.exp(), weight)
                    loss_sim_g2s = loss_function['InfoNCE'](des[0], des[1], model.module.logit_scale.exp(), weight) + loss_function['InfoNCE'](des[2], des[3], model.module.logit_scale.exp(), weight)
                else:
                    loss_sim_self1 = loss_function['InfoNCE'](des[0], des[0], model.logit_scale.exp()) + loss_function['InfoNCE'](des[2], des[2], model.logit_scale.exp())
                    loss_sim_self2 = loss_function['InfoNCE'](des[1], des[1], model.logit_scale.exp()) + loss_function['InfoNCE'](des[3], des[3], model.logit_scale.exp())
                    loss_sim_g2s = loss_function['InfoNCE'](des[0], des[1], model.logit_scale.exp()) + loss_function['InfoNCE'](des[2], des[3], model.logit_scale.exp())

                recon_loss = F.mse_loss(recon[0], feats[0]) + F.mse_loss(recon[1], feats[1])
                # slot_loss = slot_infoNCE(attent[0], attent[1]) + slot_infoNCE(attent[0], attent[0]) + slot_infoNCE(attent[1], attent[1]) + train_config.reg*reg_InfoNCE(attent[0], attent[1])
                slot_loss = slot_infoNCE(attent[0], attent[1]) + train_config.reg*reg_InfoNCE(attent[0], attent[1])
                # distill_loss = relational_distillation_3d_loss(des[0], des[2]) + relational_distillation_3d_loss(des[1], des[3])
                # loss = train_config.rec*recon_loss + train_config.dis*distill_loss + 4.0*(loss_sim_self1 + loss_sim_self2 + loss_sim_g2s + slot_loss)
                loss = train_config.rec*recon_loss + 4.0*(loss_sim_self1 + loss_sim_self2 + loss_sim_g2s + slot_loss)
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def train_base(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
          
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, weight in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                # weight = weight.to(train_config.device)
            
                # Forward pass
                des = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss_sim_self1 = loss_function['InfoNCE'](des[0], des[0], model.module.logit_scale.exp(), weight)
                    loss_sim_self2 = loss_function['InfoNCE'](des[1], des[1], model.module.logit_scale.exp(), weight)
                    loss_sim_g2s = loss_function['InfoNCE'](des[0], des[1], model.module.logit_scale.exp(), weight)
                else:
                    loss_sim_self1 = loss_function['InfoNCE'](des[0], des[0], model.logit_scale.exp())
                    loss_sim_self2 = loss_function['InfoNCE'](des[1], des[1], model.logit_scale.exp())
                    loss_sim_g2s = loss_function['InfoNCE'](des[0], des[1], model.logit_scale.exp())

                loss = 4.0*(loss_sim_self1 + loss_sim_self2 + loss_sim_g2s)
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def predict(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_sat(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(img1=dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img1=img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_uav(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(img2=dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img2=img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict3d(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    dummy_input1 = torch.randn(1, *dataloader.dataset[0][1].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output = model(img1=dummy_input, pc=dummy_input1)
    output_shape = output[0].shape[1:]
    output_shape1 = output[1].shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    pc_features = torch.zeros((len(dataloader.dataset), *output_shape1), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, pc, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            pc = pc.to(train_config.device)
            img_feature, pc_feature = model(img1=img, pc=pc)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            pc_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = pc_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, pc_features, ids


def predict_vln_query(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input1 = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device).half()
    # # 取出 dataset 里一个样本的文本 ids 作为参考
    # sample_ids = dataloader.dataset[0][1]   # shape: [seq_len], dtype: LongTensor
    # seq_len = sample_ids.shape[0]

    # # 构造 dummy input: batch_size=1, vocab_size 假设 49408（OpenAI CLIP BPE 的大小）
    # vocab_size = 49408
    # dummy_input2 = torch.randint(
    #     low=0,
    #     high=vocab_size,
    #     size=(1, seq_len),              # batch=1
    #     device=train_config.device,
    #     dtype=torch.long
    # )

    # # output_shape = model(dummy_input)[0].shape[1:]
    # output_shape = model(img1=dummy_input1, text=dummy_input2).shape
    output_shape = model(dummy_input1)[0].shape
    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img1=img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_vln_gallery(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device).half()
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(img2=dummy_input)[-1].shape

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img2=img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_3view(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features1 = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    img_features2 = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    sat_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    # bev_img, query_img, gallery_img, point_img.astype(np.float32), idx, label
    with torch.no_grad(), autocast():
        for i, (img1, img2, sat_img, _, _, ids_current) in enumerate(tqdm(dataloader)):
            img1 = img1.to(train_config.device)
            img2 = img2.to(train_config.device)
            sat_img = sat_img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature1 = model(img1)
            img_feature2 = model(img2)
            sat_feature = model(sat_img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature1 = F.normalize(img_feature1, dim=-1)
                img_feature2 = F.normalize(img_feature2, dim=-1)
                sat_feature = F.normalize(sat_feature, dim=-1)

            img_features1[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature1
            img_features2[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature2
            sat_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = sat_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features1, img_features2, sat_features, ids

def predict_eval(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    imgs = torch.zeros((len(dataloader.dataset), 3, 448, 448), dtype=torch.float32, device=train_config.device)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    qnames = []
    with torch.no_grad(), autocast():
        for i, (img, ids_current, img_pth) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current
            imgs[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img
            qnames.extend(list(img_pth))

    return img_features, ids, imgs, qnames

def predict_vigor(train_config, model, dataloader):
    model.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img)

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

