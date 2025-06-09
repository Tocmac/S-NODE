import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import lmdb
import cv2
import random


class celeba_hq_dataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        np.random.seed()
        idx = np.random.randint(0, self.num_imgs)
        img_name = os.path.join(self.root_dir, '%d.jpg'%(idx))
        image = io.imread(img_name)
        image = image * 1.0 / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image



class afhq_dataset(Dataset):
    """AFHQ dataset."""

    def __init__(self, root_dir, transform=None, condition=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.condition = condition
        class_names = ['cat', 'dog', 'wild']
        if condition is not None:
            class_path = os.path.join(root_dir, class_names[condition])
            image_list = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in image_list])
            self.labels.extend([condition] * len(image_list))
        else:
            for i, class_name in enumerate(class_names):
                class_path = os.path.join(root_dir, class_name)
                image_list = os.listdir(class_path)
                self.image_paths.extend([os.path.join(class_path, img) for img in image_list])
                self.labels.extend([i] * len(image_list))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx])
        image = io.imread(image_path)

        image = image * 1.0 / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        if self.condition is not None:
            #print(self.condition)
            return image
        return image, label

class face_dataset(Dataset):
    """FACE dataset."""

    def __init__(self, root_dir, transform=None, condition=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.condition = condition
        class_names = ['celeba', 'metface']
        if condition is not None:
            class_path = os.path.join(root_dir, class_names[condition])
            image_list = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in image_list])
            self.labels.extend([condition] * len(image_list))
        else:
            for i, class_name in enumerate(class_names):
                class_path = os.path.join(root_dir, class_name)
                image_list = os.listdir(class_path)
                self.image_paths.extend([os.path.join(class_path, img) for img in image_list])
                self.labels.extend([i] * len(image_list))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx])
        image = io.imread(image_path)

        image = image * 1.0 / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        if self.condition is not None:
            #print(self.condition)
            return image
        return image, label

'''
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        class_names = ['cat', 'dog', 'wild']

        for i, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            image_list = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in image_list])
            self.labels.extend([i] * len(image_list))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        # 进行必要的预处理操作，如缩放、裁剪、归一化等
        
        # 返回图像和标签作为样本
        return image, label
'''


