from torch.utils.data import Dataset
import torch
from PIL import Image


class CustomFlowerDataSet(Dataset):
    def __init__(self, transform, image_names, labels):
        self.transform = transform
        self.total_imgs = image_names
        self.labels = labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image = Image.open(self.total_imgs[idx]).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, torch.as_tensor([self.labels[idx]])