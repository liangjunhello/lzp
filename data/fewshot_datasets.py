import math
import os

import json
import pdb
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train',noise=None, n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.noise = noise
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        print(f"数据集模式(mode): {self.mode}")
        label = self.label_list[idx]
        if self.transform and self.mode == 'train':
            image = self.transform(image)
        if  self.transform and self.mode=='test':
            print("进入测试集噪声添加逻辑")
            image = self.transform(image)
            patch=(int)(196*self.noise)
            if patch==0:
                return image, torch.tensor(label).long()
            else:
                image=self.add_noise_to_image_list(image,num_noisy_patches=patch)





        return image, torch.tensor(label).long()


    def add_noise_to_patches(self, image_tensor, num_noisy_patches, patch_size, noise_std):


        assert image_tensor.shape == (3, 224, 224), "输入图像张量的形状必须是 (3, 224, 224)"


        image_size = 224
        num_patches_per_side = image_size // patch_size
        total_patches = num_patches_per_side ** 2

        noise_indices = torch.randperm(total_patches)[:num_noisy_patches]


        patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(3, num_patches_per_side, num_patches_per_side, patch_size, patch_size)


        for idx in noise_indices:
            row = idx // num_patches_per_side
            col = idx % num_patches_per_side
            noise = torch.randn_like(patches[:, row, col]) * noise_std
            patches[:, row, col] += noise


        noisy_image = patches.permute(0, 1, 3, 2, 4).contiguous().view(3, image_size, image_size)

        return noisy_image

    def add_noise_to_image_list(self, image_list, num_noisy_patches=196, patch_size=16, noise_std=0.5):

        noisy_image_list = []
        for image_tensor in image_list:
            noisy_image = self.add_noise_to_patches(image_tensor, num_noisy_patches, patch_size, noise_std)
            noisy_image_list.append(noisy_image)
        return noisy_image_list

fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397',
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["", "data/data_splits/split_zhou_EuroSAT.json"]
}

def build_fewshot_dataset(set_id, root, transform, mode='train',noise=None,n_shot=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform, noise)
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode,noise, n_shot, transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None, noise=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        self.noise = noise

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]

        if self.transform and self.mode == 'train':
            image = self.transform(image)

        if self.transform and self.mode == 'test':
            transformed_images = self.transform(image)
            if isinstance(transformed_images, list):
                image = transformed_images[0]
            else:
                image = transformed_images

            patch = (int)(196 * self.noise) if self.noise else 0
            if patch == 0:
                return image, torch.tensor(label).long()
            else:
                image = self.add_noise_to_patches(image, num_noisy_patches=patch, patch_size=16, noise_std=0.5)
                return image, torch.tensor(label).long()

        return image, torch.tensor(label).long()

    def add_noise_to_patches(self, image_tensor, num_noisy_patches, patch_size, noise_std):
        assert image_tensor.shape == (3, 224, 224), "输入图像张量的形状必须是 (3, 224, 224)"

        image_size = 224
        num_patches_per_side = image_size // patch_size
        total_patches = num_patches_per_side ** 2

        noise_indices = torch.randperm(total_patches)[:num_noisy_patches]

        patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(3, num_patches_per_side, num_patches_per_side, patch_size, patch_size)

        for idx in noise_indices:
            row = idx // num_patches_per_side
            col = idx % num_patches_per_side
            noise = torch.randn_like(patches[:, row, col]) * noise_std
            patches[:, row, col] += noise

        noisy_image = patches.permute(0, 1, 3, 2, 4).contiguous().view(3, image_size, image_size)
        return noisy_image
