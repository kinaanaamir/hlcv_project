from torch.utils.data import Dataset
import torch
from PIL import Image
import tqdm


class CustomCaltechDataSet(Dataset):
    def __init__(self, transform, image_names, labels):
        self.transform = transform
        self.total_imgs = image_names
        self.labels = labels

        self.images = []
        for img_path in tqdm.tqdm(self.total_imgs):
            image = Image.open(img_path).convert("RGB")
            tensor_image = self.transform(image)
            self.images.append(tensor_image)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        return self.images[idx], torch.as_tensor([self.labels[idx]])