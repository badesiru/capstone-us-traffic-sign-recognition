#TrafficLightDataset - pytorch dataset loads our cropped images with csv
#each csv has filepath and lavel, and classes r red yellow green and inactive
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random

CLASSES = ["red", "yellow", "green", "inactive"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

class TrafficLightDataset(Dataset):
    def __init__(self, manifest_path, augment=False):
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)

        self.augment = augment

        #basic transforms 
        self.base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        #augs for training
        self.aug_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["filepath"])
        label = CLASS_TO_IDX[row["label"]]

        #loads image
        img = Image.open(img_path).convert("RGB")

        #pnly applied during training 
        if self.augment:
            img = self.aug_transform(img)
        else:
            img = self.base_transform(img)

        return img, label


def get_dataloaders(batch_size=64, num_workers=2):
    train_set = TrafficLightDataset("data/lisa/train_manifest.csv", augment=True)
    val_set = TrafficLightDataset("data/lisa/val_manifest.csv", augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
