import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Specialized Augmentations for Exact Matching
# ----------------------------------------


def get_light_augmentations():
    """Light augmentations for anchor images (more like original)"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_heavy_augmentations():
    """Heavy augmentations for positive images (simulating user photos)"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=20, val_shift_limit=15, p=0.6),
        A.GaussianBlur(blur_limit=5, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Perspective(scale=(0.02, 0.08), p=0.6),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=2, p=0.4),
        A.RandomRain(blur_value=2, brightness_coefficient=0.8, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Exact Matching Dataset
# ----------------------------------------


class ExactMatchingDataset(torch.utils.data.Dataset):
    """Dataset designed for exact stamp matching with different augmentation levels"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = self._create_label_indices()

        # Create different augmentation pipelines
        self.light_transform = get_light_augmentations()
        self.heavy_transform = get_heavy_augmentations()
        self.standard_transform = get_stamp_augmentations()

    def _create_label_indices(self):
        label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def _load_raw_image(self, idx):
        """Load raw image without dataset transforms"""
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        # Get raw image
        raw_image = self._load_raw_image(index)
        anchor_label = self.base_dataset.label_mapping[self.base_dataset.image_paths[index]]

        # Apply different augmentations
        anchor_img = self.light_transform(image=raw_image)[
            "image"]      # Light augmentation
        # Heavy augmentation (user photo)
        positive_img = self.heavy_transform(image=raw_image)["image"]

        # Sample negative from different stamp
        negative_labels = [
            label for label in self.label_to_indices.keys() if label != anchor_label]

        if not negative_labels:
            raise ValueError("Cannot create triplets with only one class")

        negative_label = random.choice(negative_labels)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_raw = self._load_raw_image(negative_index)
        negative_img = self.standard_transform(image=negative_raw)[
            "image"]  # Standard augmentation

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)

# ----------------------------------------
# 3. Training Function
# ----------------------------------------


def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader)}")

# ----------------------------------------
# 4. Setup + Run
# ----------------------------------------


if __name__ == "__main__":
    image_root = "./images/original"
    batch_size = 4  # Reduced for more intensive augmentations
    embedding_dim = 256  # Increased for better discrimination
    epochs = 15  # More epochs for exact matching
    lr = 5e-5  # Lower learning rate for stability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")
    print("🎯 Training for EXACT stamp matching (not just similarity)")

    # Load dataset
    base_dataset = StampDataset(
        image_root=image_root,
        transform=None  # We'll handle transforms in the triplet dataset
    )

    print(f"✅ Found {len(base_dataset)} images")

    # Use exact matching dataset
    triplet_dataset = ExactMatchingDataset(base_dataset)

    # Check dataset integrity
    print(
        f"✅ Number of unique labels: {len(triplet_dataset.label_to_indices)}")
    print(
        f"✅ Images per label: {len(base_dataset) // len(triplet_dataset.label_to_indices)}")

    train_loader = DataLoader(
        triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Model, loss, optimizer with improved parameters
    model = StampEncoder(output_dim=embedding_dim).to(device)
    # Larger margin for exact matching
    criterion = nn.TripletMarginLoss(margin=2.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=1e-5)  # Add weight decay

    # Test a single batch
    print("🔍 Testing data loading...")
    test_batch = next(iter(train_loader))
    anchor_test, positive_test, negative_test = test_batch
    print(f"   Anchor shape: {anchor_test.shape}, dtype: {anchor_test.dtype}")
    print(
        f"   Positive shape: {positive_test.shape}, dtype: {positive_test.dtype}")
    print(
        f"   Negative shape: {negative_test.shape}, dtype: {negative_test.dtype}")
    print(
        f"   Anchor range: [{anchor_test.min():.3f}, {anchor_test.max():.3f}]")

    # Train
    train(model, train_loader, criterion, optimizer, device, epochs=epochs)

    # Save model
    os.makedirs("./saved_models", exist_ok=True)
    torch.save(model.state_dict(),
               "./saved_models/stamp_encoder_exact_matching.pth")
    print("✅ Model saved to ./saved_models/stamp_encoder_exact_matching.pth")
