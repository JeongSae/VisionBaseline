import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch.utils import data
from torchvision import datasets, models, transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import io
import os
import cv2

class ImageFolder(data.Dataset):
    def __init__(self, input_dir, target_dir, setid, image_size=224, mode='train'):
        # initalization
        self.img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
        self.labels = io.loadmat(target_dir)['labels'][0] - 1
        
        if mode == 'train':
            self.setid = io.loadmat(setid)['trnid'][0]
        elif mode == 'valid':
            self.setid = io.loadmat(setid)['valid'][0]
        else:
            self.setid = io.loadmat(setid)['tstid'][0]

        # split based setid
        self.img_paths = [self.img_paths[idx-1] for idx in self.setid]
        self.labels =[self.labels[idx-1] for idx in self.setid]

        self.image_size = image_size
        self.mode = mode
        print("image count in {} path :{}".format(self.mode, len(self.img_paths)),
              "target image count in {} path :{}".format(self.mode, len(self.labels)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = cv2.imread(self.img_paths[index])
        label = self.labels[index]
        
        # Declare an augmentation pipeline
        transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # preprocessing
        augmentations = transform(image=image)
        augmentation_img = augmentations["image"]
        
        return augmentation_img, label

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)