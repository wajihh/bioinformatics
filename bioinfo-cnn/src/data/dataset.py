import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class BacterialColonyDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset='train'):
        """
        Args:
            root_dir (str): Directory with all the bacterial colony images
            transform (callable, optional): Optional transform to be applied on a sample
            subset (str): 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.subset = subset
        
        # Get all class directories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.tif'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Could not read image {img_path}: {e}. Skipping.")
            # Try next image (wraps around if at end)
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, label

def get_transforms(subset='train'):
    """Get data transforms for training and validation"""
    if subset == 'train':
        return A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 