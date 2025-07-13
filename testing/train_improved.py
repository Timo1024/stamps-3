import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from data.train_dataset import StampDataset
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Improved Augmentations
# ----------------------------------------


def get_clean_augmentations():
    """Minimal augmentations for clean reference images"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_moderate_augmentations():
    """Moderate augmentations for positive pairs"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=15, val_shift_limit=10, p=0.6),
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.Perspective(scale=(0.02, 0.05), p=0.5),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.4),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_negative_augmentations():
    """Standard augmentations for negative samples"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=10, val_shift_limit=8, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Better Dataset Class
# ----------------------------------------


class ImprovedStampDataset(torch.utils.data.Dataset):
    """Improved dataset focusing on better discriminative learning"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = self._create_label_indices()

        # Create augmentation pipelines
        self.clean_transform = get_clean_augmentations()
        self.moderate_transform = get_moderate_augmentations()
        self.negative_transform = get_negative_augmentations()

        print(
            f"📊 Dataset created with {len(self.label_to_indices)} unique stamps")

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

    def __getitem__(self, index):
        # Get raw image
        raw_image = self._load_raw_image(index)
        anchor_label = self.base_dataset.label_mapping[self.base_dataset.image_paths[index]]

        # Create anchor (clean version)
        anchor_img = self.clean_transform(image=raw_image)["image"]

        # Create positive (moderate augmentation of same image)
        positive_img = self.moderate_transform(image=raw_image)["image"]

        # Sample negative from different stamp
        negative_labels = [
            label for label in self.label_to_indices.keys() if label != anchor_label]
        if not negative_labels:
            raise ValueError("Cannot create triplets with only one class")

        negative_label = random.choice(negative_labels)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_raw = self._load_raw_image(negative_index)
        negative_img = self.negative_transform(image=negative_raw)["image"]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)

# ----------------------------------------
# 3. Improved Loss Function
# ----------------------------------------


class ImprovedTripletLoss(nn.Module):
    """Improved triplet loss with additional constraints"""

    def __init__(self, margin=1.0, mining_strategy='batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Basic triplet loss
        basic_loss = F.relu(pos_dist - neg_dist + self.margin)

        # Additional constraints
        # 1. Positive pairs should be close (< 0.5)
        pos_penalty = F.relu(pos_dist - 0.5)

        # 2. Negative pairs should be far (> 1.5)
        neg_penalty = F.relu(1.5 - neg_dist)

        # 3. Encourage embedding diversity (prevent collapse)
        anchor_norm = torch.norm(anchor, p=2, dim=1)
        # Allow some deviation from unit norm
        norm_penalty = F.relu(torch.abs(anchor_norm - 1.0) - 0.1)

        total_loss = basic_loss.mean() + 0.2 * pos_penalty.mean() + 0.2 * \
            neg_penalty.mean() + 0.1 * norm_penalty.mean()

        return total_loss, basic_loss.mean(), pos_penalty.mean(), neg_penalty.mean()

# ----------------------------------------
# 4. Training Function with Better Monitoring
# ----------------------------------------


def train_improved(model, train_loader, criterion, optimizer, scheduler, device, epochs=20):
    """Improved training with better monitoring and validation"""

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        running_total_loss = 0.0
        running_triplet_loss = 0.0
        running_pos_penalty = 0.0
        running_neg_penalty = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Get embeddings
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            # Normalize embeddings (crucial for good similarity computation)
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
            running_total_loss += total_loss.item()
            running_triplet_loss += triplet_loss.item()
            running_pos_penalty += pos_penalty.item()
            running_neg_penalty += neg_penalty.item()

            # Update progress bar
            pbar.set_postfix({
                'total': running_total_loss / (batch_idx + 1),
                'triplet': running_triplet_loss / (batch_idx + 1),
                'pos_pen': running_pos_penalty / (batch_idx + 1),
                'neg_pen': running_neg_penalty / (batch_idx + 1)
            })

        # Epoch summary
        epoch_loss = running_total_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(
            f"  Triplet Loss: {running_triplet_loss / len(train_loader):.4f}")
        print(
            f"  Positive Penalty: {running_pos_penalty / len(train_loader):.4f}")
        print(
            f"  Negative Penalty: {running_neg_penalty / len(train_loader):.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Update learning rate
        scheduler.step()

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       "./saved_models/stamp_encoder_best.pth")
            print(f"  💾 New best model saved (loss: {best_loss:.4f})")

        # Validation every few epochs
        if (epoch + 1) % 5 == 0:
            print("\n🔍 Running mini-validation...")
            validate_model(model, train_loader, device, num_samples=20)


def validate_model(model, train_loader, device, num_samples=20):
    """Quick validation to check if model is learning discriminative features"""
    model.eval()

    # Get some samples
    sample_data = []
    for i, (anchor, positive, negative) in enumerate(train_loader):
        if i >= num_samples:
            break
        sample_data.append((anchor, positive, negative))

    similarities_pos = []
    similarities_neg = []

    with torch.no_grad():
        for anchor, positive, negative in sample_data:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = F.normalize(model(anchor), p=2, dim=1)
            positive_out = F.normalize(model(positive), p=2, dim=1)
            negative_out = F.normalize(model(negative), p=2, dim=1)

            # Calculate similarities
            pos_sim = F.cosine_similarity(anchor_out, positive_out, dim=1)
            neg_sim = F.cosine_similarity(anchor_out, negative_out, dim=1)

            similarities_pos.extend(pos_sim.cpu().tolist())
            similarities_neg.extend(neg_sim.cpu().tolist())

    pos_mean = np.mean(similarities_pos)
    neg_mean = np.mean(similarities_neg)
    separation = pos_mean - neg_mean

    print(
        f"  Positive similarities: {pos_mean:.4f} ± {np.std(similarities_pos):.4f}")
    print(
        f"  Negative similarities: {neg_mean:.4f} ± {np.std(similarities_neg):.4f}")
    print(f"  Separation: {separation:.4f}")

    if separation > 0.3:
        print("  ✅ Good separation!")
    elif separation > 0.1:
        print("  ⚠️  Moderate separation")
    else:
        print("  ❌ Poor separation - model not learning well")

    model.train()

# ----------------------------------------
# 5. Main Training Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    batch_size = 16  # Reasonable batch size
    embedding_dim = 256  # Smaller embedding for better training
    epochs = 30
    initial_lr = 1e-3  # Higher initial learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")
    print(f"🎯 Training IMPROVED stamp matching model")
    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"🔄 Epochs: {epochs}")

    # Load dataset
    base_dataset = StampDataset(image_root=image_root, transform=None)
    print(f"✅ Found {len(base_dataset)} images")

    # Create improved dataset
    improved_dataset = ImprovedStampDataset(base_dataset)

    train_loader = DataLoader(
        improved_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Model setup
    model = StampEncoder(output_dim=embedding_dim).to(device)

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    # Improved loss function
    criterion = ImprovedTripletLoss(margin=1.2)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    # Test data loading
    print("🔍 Testing data loading...")
    test_batch = next(iter(train_loader))
    anchor_test, positive_test, negative_test = test_batch
    print(
        f"   Batch shapes: {anchor_test.shape}, {positive_test.shape}, {negative_test.shape}")

    # Train
    print("🚀 Starting improved training...")
    train_improved(model, train_loader, criterion,
                   optimizer, scheduler, device, epochs)

    # Save final model
    os.makedirs("./saved_models", exist_ok=True)
    final_path = "./saved_models/stamp_encoder_improved.pth"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Final model saved to {final_path}")

    # Final validation
    print("\n🎯 Final validation:")
    validate_model(model, train_loader, device, num_samples=50)


if __name__ == "__main__":
    main()
