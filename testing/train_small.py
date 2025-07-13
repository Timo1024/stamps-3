import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from data.train_dataset import StampDataset, Ten_Classes_Dataset
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Augmentations for Small Dataset
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
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Small Dataset Class
# ----------------------------------------


class TenClassStampDataset(torch.utils.data.Dataset):
    """Dataset for training with individual stamps"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.num_classes = len(base_dataset)

        # Create augmentation pipelines
        self.clean_transform = get_clean_augmentations()
        self.moderate_transform = get_moderate_augmentations()
        self.negative_transform = get_negative_augmentations()

        print(
            f"📊 Individual stamps dataset created with {self.num_classes} stamps")

    def _load_raw_image(self, idx):
        """Load raw image without transforms"""
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        # For small dataset, cycle through all images multiple times per epoch
        actual_idx = index % self.num_classes
        raw_image = self._load_raw_image(actual_idx)
        anchor_label = actual_idx

        # Create anchor (clean version)
        anchor_img = self.clean_transform(image=raw_image)["image"]

        # Create positive (moderate augmentation of same image)
        positive_img = self.moderate_transform(image=raw_image)["image"]

        # Sample negative from different class
        negative_options = [i for i in range(
            self.num_classes) if i != actual_idx]
        if not negative_options:
            # Fallback: use same image but with different augmentation
            negative_idx = actual_idx
        else:
            negative_idx = random.choice(negative_options)

        negative_raw = self._load_raw_image(negative_idx)
        negative_img = self.negative_transform(image=negative_raw)["image"]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        # Return a larger number to create more training examples per epoch
        return self.num_classes * 10  # 10 variations per class per epoch

# ----------------------------------------
# 3. Improved Loss Function
# ----------------------------------------


class SmallDatasetTripletLoss(nn.Module):
    """Triplet loss optimized for small datasets"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Basic triplet loss
        basic_loss = F.relu(pos_dist - neg_dist + self.margin)

        # Additional constraints for small dataset
        # 1. Positive pairs should be very close
        pos_penalty = F.relu(pos_dist - 0.3)

        # 2. Negative pairs should be far
        neg_penalty = F.relu(1.0 - neg_dist)

        total_loss = basic_loss.mean() + 0.5 * pos_penalty.mean() + \
            0.3 * neg_penalty.mean()

        return total_loss, basic_loss.mean(), pos_penalty.mean(), neg_penalty.mean()

# ----------------------------------------
# 4. Training Function
# ----------------------------------------


def train_small_dataset(model, train_loader, criterion, optimizer, scheduler, device, epochs=50):
    """Training function optimized for small datasets"""

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        running_total_loss = 0.0
        running_triplet_loss = 0.0
        running_pos_penalty = 0.0
        running_neg_penalty = 0.0

        # For small dataset, we can afford to be more verbose
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
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
            running_total_loss += total_loss.item()
            running_triplet_loss += triplet_loss.item()
            running_pos_penalty += pos_penalty.item()
            running_neg_penalty += neg_penalty.item()

        # Epoch summary
        num_batches = len(train_loader)
        epoch_loss = running_total_loss / num_batches

        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Triplet: {running_triplet_loss / num_batches:.4f}")
        print(f"  Pos Penalty: {running_pos_penalty / num_batches:.4f}")
        print(f"  Neg Penalty: {running_neg_penalty / num_batches:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Update learning rate
        scheduler.step()

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       "./saved_models/stamp_encoder_small_best.pth")
            print(f"  💾 New best model saved (loss: {best_loss:.4f})")

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\n🔍 Running validation...")
            validate_small_model(model, train_loader, device)


def validate_small_model(model, train_loader, device):
    """Validation function for small dataset"""
    model.eval()

    # Collect all embeddings
    all_anchors = []
    all_positives = []
    all_negatives = []

    with torch.no_grad():
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = F.normalize(model(anchor), p=2, dim=1)
            positive_out = F.normalize(model(positive), p=2, dim=1)
            negative_out = F.normalize(model(negative), p=2, dim=1)

            all_anchors.append(anchor_out)
            all_positives.append(positive_out)
            all_negatives.append(negative_out)

    # Concatenate all embeddings
    all_anchors = torch.cat(all_anchors, dim=0)
    all_positives = torch.cat(all_positives, dim=0)
    all_negatives = torch.cat(all_negatives, dim=0)

    # Calculate similarities
    pos_similarities = F.cosine_similarity(all_anchors, all_positives, dim=1)
    neg_similarities = F.cosine_similarity(all_anchors, all_negatives, dim=1)

    pos_mean = pos_similarities.mean().item()
    neg_mean = neg_similarities.mean().item()
    separation = pos_mean - neg_mean

    print(
        f"  Positive similarities: {pos_mean:.4f} ± {pos_similarities.std().item():.4f}")
    print(
        f"  Negative similarities: {neg_mean:.4f} ± {neg_similarities.std().item():.4f}")
    print(f"  Separation: {separation:.4f}")

    if separation > 0.3:
        print("  ✅ Excellent separation!")
    elif separation > 0.2:
        print("  ✅ Good separation!")
    elif separation > 0.1:
        print("  ⚠️  Moderate separation")
    else:
        print("  ❌ Poor separation")

    model.train()

# ----------------------------------------
# 5. Testing Function
# ----------------------------------------


def test_ten_class_model(model, base_dataset, device):
    """Test the individual stamps model performance"""
    print("\n🎯 TESTING INDIVIDUAL STAMPS MODEL:")
    print("=" * 50)

    clean_transform = get_clean_augmentations()
    test_transform = get_moderate_augmentations()

    model.eval()
    class_names = base_dataset.get_class_names()

    # Generate reference embeddings (one per class)
    reference_embeddings = []
    with torch.no_grad():
        for idx in range(len(base_dataset)):
            image_path = base_dataset.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            clean_img = clean_transform(image=image)[
                "image"].unsqueeze(0).to(device)
            ref_embedding = F.normalize(
                model(clean_img), p=2, dim=1).cpu().squeeze()
            reference_embeddings.append(ref_embedding)

    # Test with augmented versions
    correct_matches = 0
    total_tests = 0

    for query_idx in range(len(base_dataset)):
        image_path = base_dataset.image_paths[query_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate multiple test versions
        for test_num in range(5):  # More tests per image
            test_img = test_transform(image=image)[
                "image"].unsqueeze(0).to(device)

            with torch.no_grad():
                query_embedding = F.normalize(
                    model(test_img), p=2, dim=1).cpu().squeeze()

            # Calculate similarities with all references
            similarities = []
            for ref_idx, ref_embedding in enumerate(reference_embeddings):
                sim = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    ref_embedding.unsqueeze(0)
                ).item()
                similarities.append((sim, ref_idx))

            # Sort by similarity
            similarities.sort(reverse=True, key=lambda x: x[0])
            predicted_idx = similarities[0][1]

            is_correct = (predicted_idx == query_idx)
            correct_matches += is_correct
            total_tests += 1

            if test_num == 0:  # Only print first test for each image
                print(f"Class {query_idx} ({class_names[query_idx]}): "
                      f"Predicted {predicted_idx} ({'✓' if is_correct else '✗'})")
                print(f"  Top 3 matches: {similarities[:3]}")

    accuracy = correct_matches / total_tests if total_tests > 0 else 0
    print(
        f"\n🎯 Final Accuracy: {accuracy*100:.1f}% ({correct_matches}/{total_tests})")

    return accuracy

# ----------------------------------------
# 6. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    num_classes = 10  # Work with 10 individual stamps
    batch_size = 8   # Larger batch size for GPU
    embedding_dim = 128  # Smaller embedding for small dataset
    epochs = 50
    initial_lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Use GPU if available

    print(f"✅ Using device: {device}")
    print(f"🎯 Training on {num_classes} INDIVIDUAL STAMPS")
    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"🔄 Epochs: {epochs}")

    # Load individual stamps dataset
    base_dataset = Ten_Classes_Dataset(
        image_root=image_root, transform=None, num_classes=num_classes)
    print(f"✅ Found {len(base_dataset)} individual stamps")

    # Print selected stamps
    print("\n📋 Selected individual stamps:")
    class_names = base_dataset.get_class_names()
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")

    # Create training dataset
    train_dataset = TenClassStampDataset(base_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Use workers for faster data loading
        drop_last=False
    )

    # Model setup
    model = StampEncoder(output_dim=embedding_dim).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    # Loss function
    criterion = SmallDatasetTripletLoss(margin=1.0)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    # Train
    print("🚀 Starting individual stamps training...")
    train_small_dataset(model, train_loader, criterion,
                        optimizer, scheduler, device, epochs)

    # Save final model
    os.makedirs("./saved_models", exist_ok=True)
    final_path = "./saved_models/stamp_encoder_individual_stamps.pth"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Final model saved to {final_path}")

    # Test the model
    accuracy = test_ten_class_model(model, base_dataset, device)

    if accuracy >= 1.0:
        print("🎉 PERFECT! 100% accuracy achieved!")
        print("   Ready to scale up to full dataset")
    elif accuracy > 0.9:
        print("🎉 EXCELLENT! Model works very well")
        print("   Consider fine-tuning or ready to scale up")
    elif accuracy > 0.8:
        print("👍 GOOD! Model shows strong performance")
        print("   Consider fine-tuning before scaling up")
    else:
        print("⚠️  Model needs improvement")
        print("   Try adjusting hyperparameters or augmentations")


if __name__ == "__main__":
    main()
