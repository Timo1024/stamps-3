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


class Ten_Classes_Dataset(Dataset):
    """Dataset that uses exactly N individual stamps from all available stamps"""

    def __init__(self, image_root, transform=None, num_classes=10):
        self.image_root = image_root
        self.transform = transform

        # Find ALL individual stamps in the dataset
        all_stamps = []

        print("🔍 Scanning for all individual stamps...")

        # Walk through all countries
        for country_name in os.listdir(image_root):
            country_path = os.path.join(image_root, country_name)
            if not os.path.isdir(country_path):
                continue

            # Walk through all years
            for year_folder in os.listdir(country_path):
                year_path = os.path.join(country_path, year_folder)
                if not os.path.isdir(year_path):
                    continue

                # Walk through all sets
                for set_folder in os.listdir(year_path):
                    set_path = os.path.join(year_path, set_folder)
                    if not os.path.isdir(set_path):
                        continue

                    # Find all images in this stamp folder
                    images_in_stamp = [f for f in os.listdir(set_path)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    # Each image represents a different stamp (even if in same set)
                    for image_file in images_in_stamp:
                        image_path = os.path.join(set_path, image_file)
                        stamp_id = f"{country_name}/{year_folder}/{set_folder}/{image_file}"
                        all_stamps.append((stamp_id, image_path))

        print(f"📊 Found {len(all_stamps)} total individual stamps")

        # Sort stamps for reproducibility and take the first N
        all_stamps.sort(key=lambda x: x[0])  # Sort by stamp_id
        selected_stamps = all_stamps[:num_classes]

        # Create the dataset
        self.image_paths = []
        self.labels = []
        self.class_names = []

        for class_idx, (stamp_id, image_path) in enumerate(selected_stamps):
            self.image_paths.append(image_path)
            self.labels.append(class_idx)
            self.class_names.append(stamp_id)
            print(f"Class {class_idx}: {stamp_id}")

        print(f"Total selected: {len(self.image_paths)} individual stamps")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            # Fallback: convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_class_names(self):
        return self.class_names

    def get_reference_images(self):
        """Return the original reference images without augmentation"""
        reference_images = []
        for idx in range(len(self.image_paths)):
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Only resize and normalize, no augmentation
            basic_transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

            image = basic_transform(image=image)["image"]
            reference_images.append(image)

        return torch.stack(reference_images)
