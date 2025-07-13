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
# 1. Simple but Effective Augmentations
# ----------------------------------------


def get_reference_transform():
    """Clean transform for perfect reference images"""
    return A.Compose([
        A.Resize(256, 256),  # Smaller for faster training
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_user_photo_transform():
    """Transform to simulate user-taken photos"""
    return A.Compose([
        # Real-world variations users would have
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=20, val_shift_limit=15, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Perspective(scale=(0.02, 0.08), p=0.6),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.5),
        A.RandomShadow(p=0.3),
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_negative_transform():
    """Light augmentation for negative samples"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Simple Effective Dataset
# ----------------------------------------


class SimpleStampDataset(torch.utils.data.Dataset):
    """Simple dataset focusing on the core problem"""

    def __init__(self, image_indices, base_dataset):
        self.image_indices = image_indices
        self.base_dataset = base_dataset

        self.ref_transform = get_reference_transform()
        self.user_transform = get_user_photo_transform()
        self.neg_transform = get_negative_transform()

        print(f"📊 Simple dataset with {len(image_indices)} stamps")

    def _load_raw_image(self, idx):
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        # Get current stamp
        stamp_idx = self.image_indices[index]
        raw_image = self._load_raw_image(stamp_idx)

        # Anchor: clean reference image
        anchor = self.ref_transform(image=raw_image)["image"]

        # Positive: same stamp but augmented like user photo
        positive = self.user_transform(image=raw_image)["image"]

        # Negative: different stamp
        other_indices = [idx for idx in self.image_indices if idx != stamp_idx]
        neg_idx = random.choice(other_indices)
        neg_image = self._load_raw_image(neg_idx)
        negative = self.neg_transform(image=neg_image)["image"]

        return anchor, positive, negative

    def __len__(self):
        return len(self.image_indices)

# ----------------------------------------
# 3. Simple but Effective Loss
# ----------------------------------------


class FocusedTripletLoss(nn.Module):
    """Focused triplet loss for stamp matching"""

    def __init__(self, margin=0.8):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Basic triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)

        # Encourage positive pairs to be very close
        pos_closeness = pos_dist.mean()

        # Encourage negative pairs to be far
        neg_separation = F.relu(1.0 - neg_dist).mean()

        total_loss = triplet_loss.mean() + 0.1 * pos_closeness + 0.1 * neg_separation

        return total_loss, triplet_loss.mean(), pos_closeness, neg_separation

# ----------------------------------------
# 4. Simple Training Function
# ----------------------------------------


def train_simple(model, train_loader, criterion, optimizer, device, epochs=100):
    """Simple training focused on the core task"""

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        running_total = 0.0
        running_triplet = 0.0
        running_pos = 0.0
        running_neg = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Get embeddings and normalize them
            anchor_emb = F.normalize(model(anchor), p=2, dim=1)
            positive_emb = F.normalize(model(positive), p=2, dim=1)
            negative_emb = F.normalize(model(negative), p=2, dim=1)

            # Calculate loss
            total_loss, triplet_loss, pos_loss, neg_loss = criterion(
                anchor_emb, positive_emb, negative_emb
            )

            total_loss.backward()
            optimizer.step()

            # Update statistics
            running_total += total_loss.item()
            running_triplet += triplet_loss.item()
            running_pos += pos_loss.item()
            running_neg += neg_loss.item()

            pbar.set_postfix({
                'loss': running_total / (batch_idx + 1),
                'triplet': running_triplet / (batch_idx + 1),
                'pos': running_pos / (batch_idx + 1),
                'neg': running_neg / (batch_idx + 1)
            })

        # Epoch summary
        epoch_loss = running_total / len(train_loader)
        print(f"\nEpoch {epoch+1}: Loss={epoch_loss:.4f}")
        print(f"  Triplet: {running_triplet/len(train_loader):.4f}")
        print(f"  Positive: {running_pos/len(train_loader):.4f}")
        print(f"  Negative: {running_neg/len(train_loader):.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       "./saved_models/simple_stamp_matcher.pth")

        # Validate every 20 epochs
        if (epoch + 1) % 20 == 0:
            validate_simple(model, train_loader, device)

# ----------------------------------------
# 5. Simple Validation
# ----------------------------------------


def validate_simple(model, train_loader, device):
    """Quick validation to check learning"""
    model.eval()

    pos_sims = []
    neg_sims = []

    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(train_loader):
            if i >= 5:  # Just check a few batches
                break

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = F.normalize(model(anchor), p=2, dim=1)
            positive_emb = F.normalize(model(positive), p=2, dim=1)
            negative_emb = F.normalize(model(negative), p=2, dim=1)

            pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)

            pos_sims.extend(pos_sim.cpu().tolist())
            neg_sims.extend(neg_sim.cpu().tolist())

    pos_mean = np.mean(pos_sims)
    neg_mean = np.mean(neg_sims)
    separation = pos_mean - neg_mean

    print(f"  📊 Validation:")
    print(f"    Positive similarity: {pos_mean:.3f}")
    print(f"    Negative similarity: {neg_mean:.3f}")
    print(f"    Separation: {separation:.3f}")

    if separation > 0.4:
        print("    ✅ Excellent separation!")
    elif separation > 0.2:
        print("    👍 Good separation")
    else:
        print("    ⚠️ Needs improvement")

    model.train()

# ----------------------------------------
# 6. Testing Function
# ----------------------------------------


def test_simple_model(model, image_indices, base_dataset, device):
    """Test the model on our 10 stamps"""
    print("\n🎯 TESTING SIMPLE MODEL")
    print("=" * 50)

    model.eval()

    ref_transform = get_reference_transform()
    test_transform = get_user_photo_transform()

    # Generate embeddings for all reference images
    reference_embeddings = []
    with torch.no_grad():
        for idx in image_indices:
            image_path = base_dataset.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ref_img = ref_transform(image=image)[
                "image"].unsqueeze(0).to(device)
            embedding = F.normalize(model(ref_img), p=2, dim=1)
            reference_embeddings.append(embedding.cpu().squeeze())

    # Test each stamp with augmented versions
    correct_matches = 0
    total_tests = 0

    for test_idx, stamp_idx in enumerate(image_indices):
        image_path = base_dataset.image_paths[stamp_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Test with 3 different augmentations
        for aug_num in range(3):
            test_img = test_transform(image=image)[
                "image"].unsqueeze(0).to(device)

            with torch.no_grad():
                query_embedding = F.normalize(
                    model(test_img), p=2, dim=1).cpu().squeeze()

            # Find best match
            similarities = []
            for ref_idx, ref_emb in enumerate(reference_embeddings):
                sim = F.cosine_similarity(
                    query_embedding.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                similarities.append((sim, ref_idx))

            similarities.sort(reverse=True)
            predicted_idx = similarities[0][1]
            is_correct = (predicted_idx == test_idx)

            total_tests += 1
            if is_correct:
                correct_matches += 1

            print(
                f"Stamp {test_idx}, Aug {aug_num}: Predicted {predicted_idx} (sim={similarities[0][0]:.3f}) - {'✅' if is_correct else '❌'}")

    accuracy = correct_matches / total_tests * 100
    print(
        f"\n🎯 Final Accuracy: {accuracy:.1f}% ({correct_matches}/{total_tests})")

    if accuracy >= 90:
        print("🎉 EXCELLENT! Ready to scale up!")
    elif accuracy >= 70:
        print("👍 GOOD! Minor improvements needed")
    else:
        print("⚠️ Needs more training")

# ----------------------------------------
# 7. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    num_stamps = 10  # Start with 10 stamps
    batch_size = 4   # Small batch for intensive training
    embedding_dim = 128  # Reasonable size
    epochs = 80      # More epochs for small dataset
    lr = 1e-3        # Learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")
    print(f"🎯 Training SIMPLE but EFFECTIVE stamp matcher")
    print(f"📊 Number of stamps: {num_stamps}")
    print(f"🔄 Epochs: {epochs}")

    # Load full dataset
    base_dataset = StampDataset(image_root=image_root, transform=None)
    print(f"✅ Found {len(base_dataset)} total images")

    # Select first N stamps for training
    image_indices = list(range(min(num_stamps, len(base_dataset))))
    print(f"🔍 Using stamps at indices: {image_indices}")

    # Create simple dataset
    simple_dataset = SimpleStampDataset(image_indices, base_dataset)

    train_loader = DataLoader(
        simple_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Create model
    model = StampEncoder(output_dim=embedding_dim).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)

    # Loss and optimizer
    criterion = FocusedTripletLoss(margin=0.8)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Test data loading
    print("🔍 Testing data loading...")
    test_batch = next(iter(train_loader))
    print(f"   Batch shapes: {[x.shape for x in test_batch]}")

    # Train
    os.makedirs("./saved_models", exist_ok=True)
    print("\n🚀 Starting training...")
    train_simple(model, train_loader, criterion, optimizer, device, epochs)

    # Final test
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)

    # Load best model
    model.load_state_dict(torch.load(
        "./saved_models/simple_stamp_matcher.pth"))
    test_simple_model(model, image_indices, base_dataset, device)


if __name__ == "__main__":
    main()
