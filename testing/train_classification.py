import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random
from collections import defaultdict

# ----------------------------------------
# 1. Classification Model for Stamp Matching
# ----------------------------------------


class StampClassifier(nn.Module):
    """Classification model - simpler and more effective than triplet loss"""

    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        # Use a strong backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Add our own head
        self.embedding = nn.Linear(2048, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_embedding=False):
        # Extract features
        features = self.backbone(x)  # (batch, 2048)

        # Get embedding
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize

        if return_embedding:
            return embedding

        # Classification
        out = self.dropout(embedding)
        logits = self.classifier(out)

        return logits, embedding

# ----------------------------------------
# 2. Heavy Augmentations for Mobile Photos
# ----------------------------------------


def get_reference_transform():
    """Clean transform for reference images"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_mobile_photo_transform():
    """Heavy augmentations simulating mobile phone photos"""
    return A.Compose([
        # Lighting and color variations (most important)
        A.RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=0.9),
        A.HueSaturationValue(hue_shift_limit=30,
                             sat_shift_limit=40, val_shift_limit=30, p=0.8),
        A.RandomGamma(gamma_limit=(60, 140), p=0.8),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),

        # Camera quality effects
        A.OneOf([
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(0, 0.03), p=0.7),
        A.ImageCompression(quality_lower=40, quality_upper=95, p=0.6),

        # Geometric transformations (phone held at angles)
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
        A.Perspective(scale=(0.02, 0.15), p=0.8),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3,
                           rotate_limit=25, p=0.9),

        # Environmental effects
        A.RandomShadow(p=0.6),
        A.RandomRain(blur_value=3, brightness_coefficient=0.7, p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),

        # Final processing
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 3. Dataset Class
# ----------------------------------------


class StampDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, mode='train', samples_per_image=100):
        self.image_root = image_root
        self.mode = mode
        self.samples_per_image = samples_per_image

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(
                glob.glob(os.path.join(image_root, '**', ext), recursive=True))

        self.image_paths.sort()  # Ensure consistent ordering

        # Create label mapping
        self.label_mapping = {path: i for i,
                              path in enumerate(self.image_paths)}
        self.num_classes = len(self.image_paths)

        # Set up transforms
        if mode == 'train':
            self.transform = get_mobile_photo_transform()
        else:
            self.transform = get_reference_transform()

        print(
            f"📊 Dataset created: {len(self.image_paths)} images, {self.num_classes} classes, mode={mode}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_paths) * self.samples_per_image
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # For training, cycle through images with heavy augmentation
            image_idx = idx % len(self.image_paths)
        else:
            image_idx = idx

        image_path = self.image_paths[image_idx]
        label = self.label_mapping[image_path]

        # Load and transform image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

# ----------------------------------------
# 4. Training Function
# ----------------------------------------


def train_model(model, train_loader, val_loader, device, epochs=80):
    # Loss and optimizer
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # Warm up for 10% of training
    )

    best_accuracy = 0.0
    patience = 20
    no_improvement = 0

    print(f"🚀 Starting training for {epochs} epochs...")
    print(f"📊 Training batches per epoch: {len(train_loader)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, embeddings = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%',
                'LR': f'{current_lr:.2e}'
            })

        # Validation phase
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_accuracy = validate_model(model, val_loader, device)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                no_improvement = 0
                # Save best model
                os.makedirs("./saved_models", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_accuracy,
                    'num_classes': model.classifier.out_features
                }, './saved_models/stamp_classifier_best.pth')
                print(f"✅ New best model saved! Accuracy: {val_accuracy:.2f}%")
            else:
                no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break

    return model


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📊 Validation Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

# ----------------------------------------
# 5. Testing Function with Similarity Matching
# ----------------------------------------


def test_stamp_matching(model, image_paths, device, num_tests_per_stamp=10):
    """Test the model's ability to match augmented images to original stamps"""
    model.eval()

    reference_transform = get_reference_transform()
    test_transform = get_mobile_photo_transform()

    print("\n🎯 TESTING STAMP MATCHING (Similarity-based):")
    print("="*70)

    # Generate reference embeddings for all stamps
    reference_embeddings = []
    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = reference_transform(
                image=image)["image"].unsqueeze(0).to(device)

            embedding = model(image_tensor, return_embedding=True)
            reference_embeddings.append(embedding)

    reference_embeddings = torch.cat(reference_embeddings, dim=0)

    # Test matching
    total_tests = 0
    correct_matches = 0
    per_stamp_results = defaultdict(list)

    for stamp_idx, image_path in enumerate(image_paths):
        stamp_correct = 0
        stamp_similarities = []

        for test_num in range(num_tests_per_stamp):
            # Load and heavily augment the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test_image = test_transform(image=image)[
                "image"].unsqueeze(0).to(device)

            with torch.no_grad():
                test_embedding = model(test_image, return_embedding=True)

                # Calculate similarities with all reference embeddings
                similarities = F.cosine_similarity(
                    test_embedding, reference_embeddings, dim=1)

                # Find best match
                best_match_idx = torch.argmax(similarities).item()
                similarity_score = similarities[best_match_idx].item()

                is_correct = (best_match_idx == stamp_idx)
                if is_correct:
                    stamp_correct += 1
                    correct_matches += 1

                stamp_similarities.append(similarity_score)
                per_stamp_results[stamp_idx].append({
                    'correct': is_correct,
                    'similarity': similarity_score,
                    'predicted': best_match_idx
                })

                total_tests += 1

        stamp_accuracy = stamp_correct / num_tests_per_stamp * 100
        avg_similarity = np.mean(stamp_similarities)
        print(f"Stamp {stamp_idx:2d}: {stamp_accuracy:5.1f}% ({stamp_correct:2d}/{num_tests_per_stamp}) | Sim: {avg_similarity:.3f}")

    overall_accuracy = correct_matches / total_tests * 100
    print(
        f"\n🏆 OVERALL ACCURACY: {overall_accuracy:.1f}% ({correct_matches}/{total_tests})")

    # Show detailed analysis
    perfect_stamps = sum(1 for results in per_stamp_results.values()
                         if all(r['correct'] for r in results))
    print(
        f"📈 Perfect stamps (100% accuracy): {perfect_stamps}/{len(image_paths)}")

    # Show confusion analysis
    print(f"\n🔍 CONFUSION ANALYSIS:")
    confusion_counts = defaultdict(int)
    for stamp_idx, results in per_stamp_results.items():
        for result in results:
            if not result['correct']:
                confusion_counts[(stamp_idx, result['predicted'])] += 1

    if confusion_counts:
        print("Most common confusions:")
        sorted_confusions = sorted(
            confusion_counts.items(), key=lambda x: x[1], reverse=True)
        for (true_idx, pred_idx), count in sorted_confusions[:5]:
            print(f"  Stamp {true_idx} → Stamp {pred_idx}: {count} times")

    # Recommendations
    if overall_accuracy < 90:
        print("\n💡 RECOMMENDATIONS:")
        print("- Train for more epochs")
        print("- Try different backbone architectures (EfficientNet, ViT)")
        print("- Experiment with different augmentation strategies")
        print("- Consider ensemble methods")

    return overall_accuracy

# ----------------------------------------
# 6. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    epochs = 80
    embedding_dim = 512

    print(f"🔧 Using device: {device}")
    print("🎯 Training CLASSIFICATION-based stamp matching model")
    print(f"📊 Target epochs: {epochs}")

    # Create datasets
    train_dataset = StampDataset(
        image_root, mode='train', samples_per_image=100)
    val_dataset = StampDataset(image_root, mode='val')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"🗂️ Training samples: {len(train_dataset)}")
    print(f"🗂️ Validation samples: {len(val_dataset)}")

    # Create model
    model = StampClassifier(
        num_classes=train_dataset.num_classes,
        embedding_dim=embedding_dim
    ).to(device)

    print(
        f"🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    model = train_model(model, train_loader, val_loader, device, epochs)

    # Test the model using similarity matching
    accuracy = test_stamp_matching(model, train_dataset.image_paths, device)

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_accuracy': accuracy,
        'num_classes': model.classifier.out_features,
        'embedding_dim': embedding_dim
    }, './saved_models/stamp_classifier_final.pth')

    print(f"\n💾 Final model saved!")

    # Results summary
    if accuracy >= 95:
        print("🎉 OUTSTANDING! Model achieved ≥95% accuracy!")
        print("🚀 Ready for production use!")
    elif accuracy >= 85:
        print("✅ EXCELLENT! Model achieved ≥85% accuracy!")
        print("🔧 Consider fine-tuning for even better results")
    elif accuracy >= 70:
        print("👍 GOOD! Model achieved ≥70% accuracy!")
        print("🔧 Try more training or architecture improvements")
    else:
        print("⚠️ Model needs significant improvement")


if __name__ == "__main__":
    main()
