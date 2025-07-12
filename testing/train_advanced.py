import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Enhanced Augmentations for Real-World Conditions
# ----------------------------------------


def get_reference_augmentations():
    """Minimal augmentations for reference images (clean, catalog-like)"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=2, sat_shift_limit=5, val_shift_limit=5, p=0.2),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_user_photo_augmentations():
    """Aggressive augmentations simulating user-taken photos"""
    return A.Compose([
        # Lighting and exposure variations
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30, val_shift_limit=25, p=0.8),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),

        # Blur and noise (camera shake, poor focus)
        A.GaussianBlur(blur_limit=5, p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(0, 0.02), p=0.6),

        # Geometric transformations (angle, perspective)
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
        A.Perspective(scale=(0.02, 0.1), p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=15, p=0.8),

        # Environmental effects
        A.RandomShadow(p=0.4),
        A.RandomRain(blur_value=2, brightness_coefficient=0.8, p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),

        # Color and compression artifacts
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.7),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),

        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_negative_augmentations():
    """Standard augmentations for negative samples"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Advanced Dataset with Hard Negative Mining
# ----------------------------------------


class AdvancedStampDataset(torch.utils.data.Dataset):
    """Advanced dataset with hard negative mining and balanced sampling"""

    def __init__(self, base_dataset, embedding_cache=None):
        self.base_dataset = base_dataset
        self.label_to_indices = self._create_label_indices()
        self.embedding_cache = embedding_cache or {}

        # Create augmentation pipelines
        self.reference_transform = get_reference_augmentations()
        self.user_transform = get_user_photo_augmentations()
        self.negative_transform = get_negative_augmentations()

        # For hard negative mining
        self.hard_negatives = {}
        self.update_frequency = 100  # Update hard negatives every N samples
        self.sample_count = 0

    def _create_label_indices(self):
        label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def _load_raw_image(self, idx):
        """Load raw image without transforms"""
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _get_hard_negative(self, anchor_label, anchor_embedding=None):
        """Get hard negative sample (most similar to anchor from different class)"""
        if anchor_embedding is None or len(self.embedding_cache) < 10:
            # Fallback to random negative
            return self._get_random_negative(anchor_label)

        # Find most similar negative sample
        max_similarity = -1
        best_negative_idx = None

        for label, indices in self.label_to_indices.items():
            if label == anchor_label:
                continue

            for idx in indices:
                if idx in self.embedding_cache:
                    similarity = F.cosine_similarity(
                        anchor_embedding.unsqueeze(0),
                        self.embedding_cache[idx].unsqueeze(0)
                    ).item()

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_negative_idx = idx

        if best_negative_idx is not None:
            return best_negative_idx
        else:
            return self._get_random_negative(anchor_label)

    def _get_random_negative(self, anchor_label):
        """Get random negative sample"""
        negative_labels = [label for label in self.label_to_indices.keys()
                           if label != anchor_label]
        if not negative_labels:
            raise ValueError("Cannot create triplets with only one class")

        negative_label = random.choice(negative_labels)
        return random.choice(self.label_to_indices[negative_label])

    def __getitem__(self, index):
        # Load raw image and get label
        raw_image = self._load_raw_image(index)
        anchor_label = self.base_dataset.label_mapping[self.base_dataset.image_paths[index]]

        # Create anchor (reference-like) and positive (user photo-like) from same image
        anchor_img = self.reference_transform(image=raw_image)["image"]
        positive_img = self.user_transform(image=raw_image)["image"]

        # Get negative sample (hard negative with probability 0.7, random otherwise)
        anchor_embedding = self.embedding_cache.get(index)
        if random.random() < 0.7 and anchor_embedding is not None:
            negative_idx = self._get_hard_negative(
                anchor_label, anchor_embedding)
        else:
            negative_idx = self._get_random_negative(anchor_label)

        negative_raw = self._load_raw_image(negative_idx)
        negative_img = self.negative_transform(image=negative_raw)["image"]

        self.sample_count += 1

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)

# ----------------------------------------
# 3. Advanced Training with Multiple Losses
# ----------------------------------------


class CombinedLoss(nn.Module):
    """Combined loss function with triplet and center loss"""

    def __init__(self, triplet_margin=2.0, center_loss_weight=0.5):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(
            margin=triplet_margin, p=2, reduction='mean')
        self.center_loss_weight = center_loss_weight

    def forward(self, anchor, positive, negative):
        # Triplet loss
        triplet_loss = self.triplet_loss(anchor, positive, negative)

        # Additional contrastive component
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # We want positive pairs to be very close (penalty for distance > 0.5)
        pos_penalty = torch.mean(torch.relu(pos_dist - 0.5))

        # We want negative pairs to be far (penalty for distance < 2.0)
        neg_penalty = torch.mean(torch.relu(2.0 - neg_dist))

        total_loss = triplet_loss + self.center_loss_weight * \
            (pos_penalty + neg_penalty)

        return total_loss, triplet_loss, pos_penalty, neg_penalty

# ----------------------------------------
# 4. Advanced Training Function
# ----------------------------------------


def train_advanced(model, train_loader, criterion, optimizer, scheduler, device, epochs=20):
    """Advanced training with embedding cache updates"""
    model.train()
    embedding_cache = {}

    for epoch in range(epochs):
        running_loss = 0.0
        running_triplet = 0.0
        running_pos_penalty = 0.0
        running_neg_penalty = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Get embeddings
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            # Normalize embeddings
            anchor_out = F.normalize(anchor_out, p=2, dim=1)
            positive_out = F.normalize(positive_out, p=2, dim=1)
            negative_out = F.normalize(negative_out, p=2, dim=1)

            # Calculate loss
            total_loss, triplet_loss, pos_penalty, neg_penalty = criterion(
                anchor_out, positive_out, negative_out
            )

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update running statistics
            running_loss += total_loss.item()
            running_triplet += triplet_loss.item()
            running_pos_penalty += pos_penalty.item()
            running_neg_penalty += neg_penalty.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'triplet': running_triplet / (batch_idx + 1),
                'pos_pen': running_pos_penalty / (batch_idx + 1),
                'neg_pen': running_neg_penalty / (batch_idx + 1)
            })

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{epochs}]:")
        print(f"  Total Loss: {running_loss / len(train_loader):.4f}")
        print(f"  Triplet Loss: {running_triplet / len(train_loader):.4f}")
        print(
            f"  Positive Penalty: {running_pos_penalty / len(train_loader):.4f}")
        print(
            f"  Negative Penalty: {running_neg_penalty / len(train_loader):.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Update embedding cache every few epochs for hard negative mining
        if (epoch + 1) % 3 == 0:
            print("🔄 Updating embedding cache for hard negative mining...")
            embedding_cache = update_embedding_cache(
                model, train_loader.dataset, device)
            train_loader.dataset.embedding_cache = embedding_cache


def update_embedding_cache(model, dataset, device):
    """Update embedding cache for hard negative mining"""
    model.eval()
    cache = {}

    with torch.no_grad():
        for idx in range(len(dataset)):
            raw_image = dataset._load_raw_image(idx)
            ref_image = dataset.reference_transform(image=raw_image)["image"]
            ref_image = ref_image.unsqueeze(0).to(device)

            embedding = model(ref_image)
            embedding = F.normalize(embedding, p=2, dim=1)
            cache[idx] = embedding.cpu().squeeze()

    model.train()
    return cache

# ----------------------------------------
# 5. Main Training Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    batch_size = 6  # Smaller batch for more intensive training
    embedding_dim = 512  # Larger embedding space
    epochs = 25  # More epochs
    initial_lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")
    print("🎯 Training ADVANCED exact stamp matching model")
    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"🔄 Epochs: {epochs}")

    # Load dataset
    base_dataset = StampDataset(
        image_root=image_root,
        transform=None  # We'll handle transforms in the advanced dataset
    )

    print(f"✅ Found {len(base_dataset)} images")

    # Create advanced dataset
    advanced_dataset = AdvancedStampDataset(base_dataset)

    # Check dataset integrity
    print(
        f"✅ Number of unique labels: {len(advanced_dataset.label_to_indices)}")

    train_loader = DataLoader(
        advanced_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Model setup
    model = StampEncoder(output_dim=embedding_dim).to(device)

    # Advanced loss function
    criterion = CombinedLoss(triplet_margin=2.5, center_loss_weight=0.3)

    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # Test data loading
    print("🔍 Testing data loading...")
    test_batch = next(iter(train_loader))
    anchor_test, positive_test, negative_test = test_batch
    print(f"   Anchor shape: {anchor_test.shape}")
    print(f"   Positive shape: {positive_test.shape}")
    print(f"   Negative shape: {negative_test.shape}")

    # Train
    print("🚀 Starting advanced training...")
    train_advanced(model, train_loader, criterion,
                   optimizer, scheduler, device, epochs)

    # Save model
    os.makedirs("./saved_models", exist_ok=True)
    model_path = "./saved_models/stamp_encoder_advanced.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Advanced model saved to {model_path}")

    # Save model with optimizer state for potential resuming
    checkpoint_path = "./saved_models/stamp_encoder_advanced_checkpoint.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'embedding_dim': embedding_dim,
    }, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
