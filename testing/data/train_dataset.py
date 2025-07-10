from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import glob


def get_stamp_augmentations():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(std_range=(0.1, 0.2), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.Perspective(scale=(0.02, 0.05), p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomRain(blur_value=1, brightness_coefficient=0.9, p=0.2),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),  # ImageNet normalization
        ToTensorV2()
    ])


class StampDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_root = image_root
        self.image_paths = glob.glob(os.path.join(
            image_root, '**', '*.*'), recursive=True)
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(
            ('.jpg', '.jpeg', '.png'))]
        self.transform = transform

        # Give each unique stamp its own label based on the image filename (or path)
        # You could also use the full relative path as the ID
        self.label_mapping = {path: i for i,
                              path in enumerate(sorted(self.image_paths))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.label_mapping[image_path]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            # Fallback: convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label
