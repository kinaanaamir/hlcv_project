from torch.utils.data import Dataset
import torch
from PIL import Image
import os


class CustomCubDataSet(Dataset):
    def __init__(self, main_dir, transform, image_names, labels):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = image_names
        self.labels = labels

        # self.images = []
        # for img_path in self.total_imgs:
        #     img_loc = os.path.join(self.main_dir, img_path)
        #     image = Image.open(img_loc).convert("RGB")
        #     tensor_image = self.transform(image)
        #     self.images.append(tensor_image)


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, torch.as_tensor([self.labels[idx]])
