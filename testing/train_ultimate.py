import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

from data.train_dataset import StampDataset

# ----------------------------------------
# 1. Improved Encoder with Better Architecture
# ----------------------------------------


class AdvancedStampEncoder(nn.Module):
    """Improved encoder with better feature extraction"""

    def __init__(self, output_dim=256):
        super().__init__()

        # Use ResNet50 as backbone (better than ResNet18)
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        # Remove the classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze early layers for better generalization
        for param in list(self.backbone.parameters())[:50]:
            param.requires_grad = False

        # Add custom head for embeddings
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return embeddings

# ----------------------------------------
# 2. Optimized Augmentations
# ----------------------------------------


def get_reference_transform():
    """Clean transform for reference images"""
    return A.Compose([
        A.Resize(384, 384),  # Higher resolution for better features
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_user_photo_transform():
    """Strong but realistic augmentations for user photos"""
    return A.Compose([
        # Lighting variations (very common in phone photos)
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.HueSaturationValue(hue_shift_limit=25,
                             sat_shift_limit=30, val_shift_limit=25, p=0.8),
        A.RandomGamma(gamma_limit=(70, 140), p=0.7),

        # Phone camera effects
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.6),
        A.GaussNoise(var_limit=(0, 0.03), p=0.7),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.6),

        # Perspective and rotation (phone held at angles)
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
        A.Perspective(scale=(0.02, 0.12), p=0.8),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2,
                           rotate_limit=20, p=0.8),

        # Environmental effects
        A.RandomShadow(p=0.5),
        A.RandomRain(blur_value=2, brightness_coefficient=0.8, p=0.3),

        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 3. Advanced Dataset with Better Sampling
# ----------------------------------------


class AdvancedStampDataset(torch.utils.data.Dataset):
    """Advanced dataset with better negative sampling"""

    def __init__(self, base_dataset, num_stamps=10):
        self.base_dataset = base_dataset
        self.num_stamps = min(num_stamps, len(base_dataset))
        self.image_indices = list(range(self.num_stamps))

        self.reference_transform = get_reference_transform()
        self.user_photo_transform = get_user_photo_transform()

        # Pre-load all images for faster training
        self.images = []
        for i in range(self.num_stamps):
            image_path = self.base_dataset.image_paths[i]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)

        print(f"✅ Advanced dataset created with {self.num_stamps} stamps")

    def __getitem__(self, index):
        # Get anchor image index
        anchor_idx = index % self.num_stamps

        # Get raw image
        raw_image = self.images[anchor_idx]

        # Create anchor (clean reference)
        anchor = self.reference_transform(image=raw_image.copy())["image"]

        # Create positive (heavily augmented version)
        positive = self.user_photo_transform(image=raw_image.copy())["image"]

        # Create hard negative (different stamp, but also augmented to be harder)
        negative_idx = random.choice(
            [i for i in self.image_indices if i != anchor_idx])
        negative_raw = self.images[negative_idx]
        negative = self.reference_transform(image=negative_raw.copy())["image"]

        return anchor, positive, negative, anchor_idx

    def __len__(self):
        return self.num_stamps * 50  # 50 samples per stamp per epoch

# ----------------------------------------
# 4. Advanced Loss Function
# ----------------------------------------


class AdaptiveTripletLoss(nn.Module):
    """Adaptive triplet loss with dynamic margin"""

    def __init__(self, base_margin=1.0, alpha=0.1):
        super().__init__()
        self.base_margin = base_margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Adaptive margin based on negative distance
        adaptive_margin = self.base_margin + self.alpha * neg_dist.detach()

        # Triplet loss with adaptive margin
        loss = F.relu(pos_dist - neg_dist + adaptive_margin)

        return loss.mean()

# ----------------------------------------
# 5. Advanced Training with Learning Rate Scheduling
# ----------------------------------------


def train_advanced(model, train_loader, criterion, optimizer, scheduler, device, epochs=100):
    """Advanced training with better monitoring"""

    model.train()
    best_separation = -999

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            # Normalize embeddings (crucial for cosine similarity)
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            negative_emb = F.normalize(negative_emb, p=2, dim=1)

            # Calculate loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{running_loss/num_batches:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        # Step scheduler
        scheduler.step()

        avg_loss = running_loss / num_batches

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            separation = validate_advanced(model, train_loader, device)

            if separation > best_separation:
                best_separation = separation
                print(f"  🎯 New best separation: {separation:.4f}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def validate_advanced(model, train_loader, device):
    """Advanced validation with better metrics"""
    model.eval()

    all_pos_sims = []
    all_neg_sims = []

    with torch.no_grad():
        for i, (anchor, positive, negative, labels) in enumerate(train_loader):
            if i >= 10:  # Use more batches for better validation
                break

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = F.normalize(model(anchor), p=2, dim=1)
            positive_emb = F.normalize(model(positive), p=2, dim=1)
            negative_emb = F.normalize(model(negative), p=2, dim=1)

            # Calculate similarities
            pos_sims = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            neg_sims = F.cosine_similarity(anchor_emb, negative_emb, dim=1)

            all_pos_sims.extend(pos_sims.cpu().numpy())
            all_neg_sims.extend(neg_sims.cpu().numpy())

    pos_mean = np.mean(all_pos_sims)
    neg_mean = np.mean(all_neg_sims)
    separation = pos_mean - neg_mean

    print(f"  📊 Validation - Pos: {pos_mean:.4f}±{np.std(all_pos_sims):.3f}, "
          f"Neg: {neg_mean:.4f}±{np.std(all_neg_sims):.3f}, "
          f"Sep: {separation:.4f}")

    model.train()
    return separation

# ----------------------------------------
# 6. Comprehensive Testing
# ----------------------------------------


def test_advanced(model, base_dataset, device, num_stamps=10):
    """Comprehensive testing with multiple metrics"""
    print("\n🎯 COMPREHENSIVE TESTING:")
    print("=" * 60)

    model.eval()

    reference_transform = get_reference_transform()
    test_transform = get_user_photo_transform()

    # Generate reference embeddings (gallery)
    reference_embeddings = []

    with torch.no_grad():
        for i in range(num_stamps):
            image_path = base_dataset.image_paths[i]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ref_tensor = reference_transform(
                image=image)["image"].unsqueeze(0).to(device)
            ref_embedding = F.normalize(model(ref_tensor), p=2, dim=1)
            reference_embeddings.append(ref_embedding)

    reference_embeddings = torch.cat(reference_embeddings, dim=0)

    # Test with multiple query variations
    total_correct = 0
    total_tests = 0
    per_stamp_accuracy = []

    for query_idx in range(num_stamps):
        stamp_correct = 0
        stamp_tests = 5  # 5 tests per stamp

        image_path = base_dataset.image_paths[query_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"\n📸 Testing stamp {query_idx}:")

        for test_num in range(stamp_tests):
            # Create heavily augmented query
            query_tensor = test_transform(
                image=image)["image"].unsqueeze(0).to(device)
            query_embedding = F.normalize(model(query_tensor), p=2, dim=1)

            # Find best match
            similarities = F.cosine_similarity(
                query_embedding, reference_embeddings, dim=1)
            predicted_idx = torch.argmax(similarities).item()
            max_similarity = similarities[predicted_idx].item()

            is_correct = (predicted_idx == query_idx)
            if is_correct:
                stamp_correct += 1
                total_correct += 1

            total_tests += 1

            status = "✅" if is_correct else "❌"
            print(
                f"  Test {test_num+1}: {status} Pred={predicted_idx}, Sim={max_similarity:.4f}")

        stamp_accuracy = stamp_correct / stamp_tests
        per_stamp_accuracy.append(stamp_accuracy)
        print(f"  🎯 Stamp {query_idx} accuracy: {stamp_accuracy*100:.1f}%")

    overall_accuracy = total_correct / total_tests

    print(f"\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS:")
    print(
        f"   Overall Accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_tests})")
    print(
        f"   Per-stamp accuracy: {[f'{acc*100:.0f}%' for acc in per_stamp_accuracy]}")
    print(
        f"   Perfect stamps: {sum(1 for acc in per_stamp_accuracy if acc == 1.0)}/{num_stamps}")

    return overall_accuracy

# ----------------------------------------
# 7. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    num_stamps = 10
    batch_size = 6  # Larger batch for better gradients
    embedding_dim = 256  # Larger embeddings
    epochs = 60
    initial_lr = 2e-4  # Lower initial learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 ADVANCED STAMP MATCHER TRAINING")
    print(f"📱 Device: {device}")
    print(f"🏷️  Stamps: {num_stamps}")
    print(f"🧠 Embedding dim: {embedding_dim}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📊 Batch size: {batch_size}")

    # Load datasets
    base_dataset = StampDataset(image_root=image_root, transform=None)
    print(f"📁 Found {len(base_dataset)} total images")

    if len(base_dataset) < num_stamps:
        print(f"⚠️  Using all {len(base_dataset)} images")
        num_stamps = len(base_dataset)

    # Create training dataset
    train_dataset = AdvancedStampDataset(base_dataset, num_stamps=num_stamps)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create advanced model
    model = AdvancedStampEncoder(output_dim=embedding_dim).to(device)
    criterion = AdaptiveTripletLoss(base_margin=1.2, alpha=0.2)
    optimizer = optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    print(
        f"🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test data loading
    print("\n🔍 Testing data loading...")
    test_batch = next(iter(train_loader))
    anchor_test, positive_test, negative_test, labels_test = test_batch
    print(f"✅ Batch shapes: {anchor_test.shape}")

    # Train
    print("\n🏋️ Starting advanced training...")
    train_advanced(model, train_loader, criterion, optimizer,
                   scheduler, device, epochs=epochs)

    # Final test
    accuracy = test_advanced(
        model, base_dataset, device, num_stamps=num_stamps)

    # Save model
    os.makedirs("./saved_models", exist_ok=True)
    model_path = "./saved_models/stamp_encoder_advanced.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'accuracy': accuracy
    }, model_path)
    print(f"💾 Model saved to {model_path}")

    # Results analysis
    if accuracy >= 0.95:
        print("🎉 OUTSTANDING! Model achieved >95% accuracy!")
        print("🚀 Ready for scaling to larger datasets!")
    elif accuracy >= 0.85:
        print("✅ EXCELLENT! Model achieved >85% accuracy!")
        print("🔧 Consider fine-tuning for even better results")
    elif accuracy >= 0.7:
        print("👍 GOOD! Model achieved >70% accuracy!")
        print("🔧 Consider more training or architecture changes")
    else:
        print("⚠️  Model needs improvement. Try:")
        print("   - More epochs")
        print("   - Different augmentations")
        print("   - Architecture changes")


if __name__ == "__main__":
    main()
