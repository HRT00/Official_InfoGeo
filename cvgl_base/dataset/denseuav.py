import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import json
import torch
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from scipy.linalg import expm, norm
import collections


class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):  
        data = torch.from_numpy(np.array(data))
        return data


class PointCloudScaling(object):
    def __init__(self, 
                 scale=[2. / 3, 3. / 2], 
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 symmetries=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.from_numpy(np.array(symmetries))
        
    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        symmetries = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.symmetries = self.symmetries.to(device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False, 
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - torch.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim+1]
                data['heights'] = height - torch.min(height)
            
            if self.centering:
                data['pos'] = data['pos'] - torch.mean(data['pos'], axis=0, keepdims=True)
            
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - torch.mean(data, axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data = data / m
        return data


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()
        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data

def get_text_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data

def get_pc_data(path):
    data = {}
    dirs = os.listdir(path)
    for name in dirs:
        data[name] = {"path": os.path.join(path, name)}
        data[name]["files"] = ["points3D.txt"] # os.path.join(data[name]["path"], "points3D.txt")

    return data

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class DenseUAVDatasetUCVGLTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 bev_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.bev_dict = get_data(bev_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            bev_img = "{}/{}".format(self.bev_dict[idx]["path"],
                                       self.bev_dict[idx]["files"][0])
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img1, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img2, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img3, "{}/{}".format(gallery_path, g)))

        # For UCVGL
        galleries = [p[-1] for p in self.pairs]   # 提取最后一列
        random.shuffle(galleries)                 # 打乱 gallery

        # 重新组合
        self.pairs = [(p[0], p[1], p[2], p[3], p[4], g) for p, g in zip(self.pairs, galleries)]

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, bev_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # pc_imgs = []
        # for pc_path in point_cloud_paths:
        #     point_img = np.loadtxt(pc_path, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        #     if self.uniform:
        #         point_img = farthest_point_sample(point_img, self.npoints)
        #     else:
        #         point_img = point_img[0:self.npoints, :]

        #     point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        #     if not self.use_rgb:
        #         point_img = point_img[:, 0:3]
        #     pc_imgs.append(point_img)
        # point_img = np.concatenate(pc_imgs, axis=0)
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        # point_img = PointsToTensor()(point_img)
        # point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
        # point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        # point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)
        if not self.use_rgb:
            point_img = point_img[:, 0:3]
        
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        bev_img = cv2.imread(bev_img_path)
        try:
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(bev_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
                bev_img = self.transforms_gallery(image=bev_img)['image']
            except TypeError as e:
                print(gallery_img_path)
                print(bev_img_path)
            
        return bev_img, query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

class DenseUAV_DatasetVLN2DTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
            except TypeError as e:
                print(gallery_img_path)

        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.2)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

class DenseUAVDatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img1, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img2, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img3, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # pc_imgs = []
        # for pc_path in point_cloud_paths:
        #     point_img = np.loadtxt(pc_path, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        #     if self.uniform:
        #         point_img = farthest_point_sample(point_img, self.npoints)
        #     else:
        #         point_img = point_img[0:self.npoints, :]

        #     point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        #     if not self.use_rgb:
        #         point_img = point_img[:, 0:3]
        #     pc_imgs.append(point_img)
        # point_img = np.concatenate(pc_imgs, axis=0)
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        # point_img = PointsToTensor()(point_img)
        # point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
        # point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        # point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)
        if not self.use_rgb:
            point_img = point_img[:, 0:3]
        
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
            except TypeError as e:
                print(gallery_img_path)
            
        return query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.2)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

class DenseUAVDatasetEvalVLN2D(Dataset):

    def __init__(self,
                 data_folder,
                 mode = 'query',
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
        self.mode = mode
        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                if self.mode == "query":
                    self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                    file))
                    self.sample_ids.append(sample_id)
                else:
                    self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                    file))
                    self.sample_ids.append(sample_id)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        sample_id = self.sample_ids[index]
        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

class DenseUAVDatasetEval_(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label, img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

class DenseUAVDatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                        A.ToSepia(p=0.5),
                                        A.Solarize(threshold=128, p=0.5),
                                      ], p=0.1),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                        A.ToSepia(p=0.5),
                                        A.Solarize(threshold=128, p=0.5),
                                    ], p=0.1),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms
