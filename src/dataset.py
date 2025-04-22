import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUTRIENTS = ["calories", "protein", "fat", "carbs"]

class Nutrition5kDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, image_extension=".png", normalize=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.normalize = normalize

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mean = self.df[NUTRIENTS].mean().values
        self.std = self.df[NUTRIENTS].std().values

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['id'] + self.image_extension)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        nutrients = row[NUTRIENTS].values.astype(np.float32)
        if self.normalize:
            nutrients = (nutrients - self.mean) / self.std

        return image_tensor, torch.tensor(nutrients)

    def __len__(self):
        return len(self.df)

    def denormalize(self, tensor):
        return tensor * torch.tensor(self.std) + torch.tensor(self.mean)

