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
# 1. Enhanced Augmentations
# ----------------------------------------


def get_clean_augmentations():
    """Minimal augmentations for clean reference images"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_light_augmentations():
    """Light augmentations for positive pairs"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=5, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_strong_augmentations():
    """Strong augmentations to simulate user photos"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.9),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=20, val_shift_limit=15, p=0.8),
        A.GaussianBlur(blur_limit=5, p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Perspective(scale=(0.02, 0.08), p=0.6),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.6),
        A.MotionBlur(blur_limit=5, p=0.3),
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
# 2. Improved Dataset with Multiple Positive Types
# ----------------------------------------


class EnhancedSmallStampDataset(torch.utils.data.Dataset):
    """Enhanced dataset with different types of positive pairs"""

    def __init__(self, image_indices, base_dataset):
        self.image_indices = image_indices
        self.base_dataset = base_dataset

        # Create augmentation pipelines
        self.clean_transform = get_clean_augmentations()
        self.light_transform = get_light_augmentations()
        self.strong_transform = get_strong_augmentations()
        self.negative_transform = get_negative_augmentations()

        print(f"📊 Enhanced dataset created with {len(image_indices)} images")

    def _load_raw_image(self, idx):
        """Load raw image without transforms"""
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        # Get actual image index (handle wraparound for extended dataset)
        actual_idx = self.image_indices[index % len(self.image_indices)]
        raw_image = self._load_raw_image(actual_idx)

        # Randomly choose anchor and positive augmentation levels
        anchor_choice = random.choice(['clean', 'light'])
        positive_choice = random.choice(['light', 'strong'])

        # Create anchor
        if anchor_choice == 'clean':
            anchor_img = self.clean_transform(image=raw_image)["image"]
        else:
            anchor_img = self.light_transform(image=raw_image)["image"]

        # Create positive (different augmentation of same image)
        if positive_choice == 'light':
            positive_img = self.light_transform(image=raw_image)["image"]
        else:
            positive_img = self.strong_transform(image=raw_image)["image"]

        # Sample negative from different image
        negative_options = [
            idx for idx in self.image_indices if idx != actual_idx]
        if not negative_options:
            # Fallback: use same image but with very different augmentation
            negative_img = self.strong_transform(image=raw_image)["image"]
        else:
            negative_idx = random.choice(negative_options)
            negative_raw = self._load_raw_image(negative_idx)
            negative_img = self.negative_transform(image=negative_raw)["image"]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        # Return more samples per epoch by repeating the dataset
        return len(self.image_indices) * 5  # 5x more training per epoch

# ----------------------------------------
# 3. Advanced Loss Function
# ----------------------------------------


class AdvancedTripletLoss(nn.Module):
    """Advanced triplet loss with adaptive margin and hard mining"""

    def __init__(self, base_margin=1.0, hard_margin=1.5):
        super().__init__()
        self.base_margin = base_margin
        self.hard_margin = hard_margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Adaptive margin based on difficulty
        # If positive is already close and negative is far, use smaller margin
        # If positive is far or negative is close, use larger margin
        difficulty = pos_dist + (2.0 - neg_dist)  # Higher = more difficult
        adaptive_margin = self.base_margin + \
            0.5 * torch.sigmoid(difficulty - 1.0)

        # Basic triplet loss with adaptive margin
        basic_loss = F.relu(pos_dist - neg_dist + adaptive_margin)

        # Hard mining: focus more on difficult triplets
        hard_triplets = basic_loss > 0.1  # Non-zero loss triplets
        if hard_triplets.any():
            hard_loss = basic_loss[hard_triplets].mean()
        else:
            hard_loss = basic_loss.mean()

        # Additional constraints
        # 1. Positive pairs should be very close
        pos_target = 0.2  # Target distance for positive pairs
        pos_penalty = F.relu(pos_dist - pos_target)

        # 2. Negative pairs should be far
        neg_target = 1.5  # Target distance for negative pairs
        neg_penalty = F.relu(neg_target - neg_dist)

        # 3. Embedding magnitude regularization
        anchor_mag = torch.norm(anchor, p=2, dim=1)
        mag_penalty = F.relu(torch.abs(anchor_mag - 1.0) - 0.1)

        # Combine losses
        total_loss = (hard_loss +
                      0.5 * pos_penalty.mean() +
                      0.3 * neg_penalty.mean() +
                      0.1 * mag_penalty.mean())

        return total_loss, basic_loss.mean(), pos_penalty.mean(), neg_penalty.mean()

# ----------------------------------------
# 4. Enhanced Training Function
# ----------------------------------------


def train_enhanced_small(model, train_loader, criterion, optimizer, scheduler, device, epochs=80):
    """Enhanced training with better monitoring"""

    model.train()
    best_loss = float('inf')
    patience = 15
    no_improvement = 0

    for epoch in range(epochs):
        running_total_loss = 0.0
        running_triplet_loss = 0.0
        running_pos_penalty = 0.0
        running_neg_penalty = 0.0

        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (anchor, positive, negative) in enumerate(epoch_bar):
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

            # Update progress bar
            epoch_bar.set_postfix({
                'loss': f"{running_total_loss / (batch_idx + 1):.4f}",
                'pos': f"{running_pos_penalty / (batch_idx + 1):.4f}",
                'neg': f"{running_neg_penalty / (batch_idx + 1):.4f}"
            })

        # Epoch summary
        num_batches = len(train_loader)
        epoch_loss = running_total_loss / num_batches

        print(f"\\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Triplet: {running_triplet_loss / num_batches:.4f}")
        print(f"  Pos Penalty: {running_pos_penalty / num_batches:.4f}")
        print(f"  Neg Penalty: {running_neg_penalty / num_batches:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Update learning rate
        scheduler.step()

        # Save best model and early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improvement = 0
            torch.save(model.state_dict(),
                       "./saved_models/stamp_encoder_enhanced_best.pth")
            print(f"  💾 New best model saved (loss: {best_loss:.4f})")
        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            print(f"  🛑 Early stopping: no improvement for {patience} epochs")
            break

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\\n🔍 Running validation...")
            validate_enhanced_model(model, train_loader, device)


def validate_enhanced_model(model, train_loader, device):
    """Enhanced validation function"""
    model.eval()

    # Collect embeddings from a subset of data
    all_anchors = []
    all_positives = []
    all_negatives = []

    sample_count = 0
    max_samples = 100  # Limit for speed

    with torch.no_grad():
        for anchor, positive, negative in train_loader:
            if sample_count >= max_samples:
                break

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = F.normalize(model(anchor), p=2, dim=1)
            positive_out = F.normalize(model(positive), p=2, dim=1)
            negative_out = F.normalize(model(negative), p=2, dim=1)

            all_anchors.append(anchor_out)
            all_positives.append(positive_out)
            all_negatives.append(negative_out)

            sample_count += anchor.size(0)

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

    # Calculate distances too
    pos_distances = F.pairwise_distance(all_anchors, all_positives, p=2)
    neg_distances = F.pairwise_distance(all_anchors, all_negatives, p=2)

    print(
        f"  Positive similarities: {pos_mean:.4f} ± {pos_similarities.std().item():.4f}")
    print(
        f"  Negative similarities: {neg_mean:.4f} ± {neg_similarities.std().item():.4f}")
    print(f"  Separation: {separation:.4f}")
    print(
        f"  Positive distances: {pos_distances.mean().item():.4f} ± {pos_distances.std().item():.4f}")
    print(
        f"  Negative distances: {neg_distances.mean().item():.4f} ± {neg_distances.std().item():.4f}")

    if separation > 0.5:
        print("  ✅ Excellent separation!")
    elif separation > 0.3:
        print("  ✅ Good separation!")
    elif separation > 0.1:
        print("  ⚠️  Moderate separation")
    else:
        print("  ❌ Poor separation")

    model.train()

# ----------------------------------------
# 5. Comprehensive Testing Function
# ----------------------------------------


def test_enhanced_model(model, image_indices, base_dataset, device):
    """Comprehensive testing of the enhanced model"""
    print("\\n🎯 TESTING ENHANCED MODEL:")
    print("=" * 60)

    clean_transform = get_clean_augmentations()
    test_transforms = [
        ("Light", get_light_augmentations()),
        ("Strong", get_strong_augmentations())
    ]

    model.eval()

    # Generate reference embeddings (clean versions)
    reference_embeddings = []
    with torch.no_grad():
        for idx in image_indices:
            image_path = base_dataset.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            clean_img = clean_transform(image=image)[
                "image"].unsqueeze(0).to(device)
            ref_embedding = F.normalize(
                model(clean_img), p=2, dim=1).cpu().squeeze()
            reference_embeddings.append(ref_embedding)

    # Test with different augmentation levels
    for test_name, test_transform in test_transforms:
        print(f"\\n📊 Testing with {test_name} augmentations:")
        print("-" * 40)

        correct_matches = 0
        total_tests = 0
        all_similarities = []

        for query_idx, actual_idx in enumerate(image_indices):
            image_path = base_dataset.image_paths[actual_idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate multiple test versions
            image_correct = 0
            image_total = 0

            for test_num in range(5):  # Test 5 versions per image
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
                top_similarity = similarities[0][0]

                is_correct = (predicted_idx == query_idx)
                correct_matches += is_correct
                total_tests += 1
                image_correct += is_correct
                image_total += 1
                all_similarities.append(top_similarity)

            accuracy_for_image = image_correct / image_total
            print(
                f"  Image {query_idx}: {accuracy_for_image*100:.0f}% ({image_correct}/{image_total})")

        overall_accuracy = correct_matches / total_tests if total_tests > 0 else 0
        mean_similarity = np.mean(all_similarities)

        print(
            f"  Overall Accuracy: {overall_accuracy*100:.1f}% ({correct_matches}/{total_tests})")
        print(f"  Mean Similarity: {mean_similarity:.4f}")

    return overall_accuracy

# ----------------------------------------
# 6. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    num_images = 10
    batch_size = 8    # Slightly larger batch
    embedding_dim = 128
    epochs = 80       # More epochs with early stopping
    initial_lr = 5e-4  # Lower initial learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")
    print(f"🎯 Training ENHANCED small dataset ({num_images} images)")
    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"🔄 Max epochs: {epochs}")

    # Load dataset and select same images for consistency
    base_dataset = StampDataset(image_root=image_root, transform=None)

    # Use same images as before for comparison
    selected_indices = [51, 209, 228, 285, 457, 501, 563, 1309, 1508, 1518]

    print(f"📋 Using same images as before: {selected_indices}")

    # Create enhanced dataset
    enhanced_dataset = EnhancedSmallStampDataset(
        selected_indices, base_dataset)

    train_loader = DataLoader(
        enhanced_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # Model setup with better initialization
    model = StampEncoder(output_dim=embedding_dim).to(device)

    # Better weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')

    model.apply(init_weights)

    # Enhanced loss function
    criterion = AdvancedTripletLoss(base_margin=0.8, hard_margin=1.2)

    # Optimizer
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

    # Train
    print("🚀 Starting enhanced training...")
    train_enhanced_small(model, train_loader, criterion,
                         optimizer, scheduler, device, epochs)

    # Load best model for testing
    if os.path.exists("./saved_models/stamp_encoder_enhanced_best.pth"):
        model.load_state_dict(torch.load(
            "./saved_models/stamp_encoder_enhanced_best.pth"))
        print("✅ Loaded best model for testing")

    # Save final model
    os.makedirs("./saved_models", exist_ok=True)
    final_path = "./saved_models/stamp_encoder_enhanced.pth"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Final model saved to {final_path}")

    # Comprehensive testing
    accuracy = test_enhanced_model(
        model, selected_indices, base_dataset, device)

    if accuracy > 0.85:
        print("\\n🎉 EXCELLENT! Model works very well on small dataset")
        print("   Ready to scale up to full dataset")
    elif accuracy > 0.7:
        print("\\n👍 GOOD! Model shows strong promise")
        print("   Consider minor fine-tuning before scaling up")
    elif accuracy > 0.5:
        print("\\n⚠️  Model shows improvement but needs work")
        print("   Continue refining on small dataset")
    else:
        print("\\n❌ Model still needs significant improvement")


if __name__ == "__main__":
    main()
