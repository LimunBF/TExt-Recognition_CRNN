import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class OCRDataset(Dataset):
    def __init__(self, label_path, img_height, img_width, label_encoder):
        self.img_height = img_height
        self.img_width = img_width
        self.label_encoder = label_encoder
        self.data = []

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                image, label = line.strip().split('\t')
                self.data.append((image, label))

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((img_height, img_width)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_encoded = self.label_encoder.encode(label.lower())
        return image, torch.tensor(label_encoded, dtype=torch.long), len(label_encoded)
