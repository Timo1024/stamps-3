import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
# 1. Simple Classification Model for 10 Stamps
# ----------------------------------------


class Simple10StampClassifier(nn.Module):
    """Simple but effective classifier for 10 stamps"""

    def __init__(self, num_classes=10, embedding_dim=256):
        super().__init__()
        # Use ResNet18 for speed (sufficient for 10 classes)
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_features=False):
        if return_features:
            # Extract features before final classification
            features = self.backbone.avgpool(self.backbone.layer4(
                self.backbone.layer3(self.backbone.layer2(
                    self.backbone.layer1(self.backbone.maxpool(
                        self.backbone.relu(self.backbone.bn1(
                            self.backbone.conv1(x)
                        ))
                    ))
                ))
            ))
            features = torch.flatten(features, 1)
            return F.normalize(features, p=2, dim=1)
        else:
            return self.backbone(x)

# ----------------------------------------
# 2. Augmentations - Heavy for Mobile Photos
# ----------------------------------------


def get_clean_transform():
    """Clean transform for reference images"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_mobile_transform():
    """Heavy augmentations simulating mobile phone photos"""
    return A.Compose([
        # Color and lighting variations (most important)
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.HueSaturationValue(hue_shift_limit=25,
                             sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RandomGamma(gamma_limit=(70, 130), p=0.7),

        # Camera effects
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(0, 0.02), p=0.6),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),

        # Geometric transformations (phone angles)
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
        A.Perspective(scale=(0.02, 0.1), p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=15, p=0.8),

        # Environmental effects
        A.RandomShadow(p=0.4),
        A.RandomRain(blur_value=2, brightness_coefficient=0.8, p=0.2),

        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 3. Dataset for 10 Stamps
# ----------------------------------------


class TenStampDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, mode='train', max_images=10, samples_per_image=50):
        self.mode = mode
        self.samples_per_image = samples_per_image

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(
                glob.glob(os.path.join(image_root, '**', ext), recursive=True))

        # Take only first N images
        self.image_paths = sorted(self.image_paths)[:max_images]

        # Create label mapping
        self.label_mapping = {path: i for i,
                              path in enumerate(self.image_paths)}
        self.num_classes = len(self.image_paths)

        # Set up transforms
        if mode == 'train':
            self.transform = get_mobile_transform()
        else:
            self.transform = get_clean_transform()

        print(
            f"📊 Dataset: {len(self.image_paths)} images, {self.num_classes} classes, mode={mode}")
        for i, path in enumerate(self.image_paths):
            print(f"  Class {i}: {os.path.basename(path)}")

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


def train_model(model, train_loader, val_loader, device, epochs=40):
    # Loss and optimizer (aggressive for fast training)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2
    )

    best_accuracy = 0.0

    print(f"🚀 Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%'
            })

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_accuracy = validate_model(model, val_loader, device)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # Save best model
                os.makedirs("./saved_models", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_accuracy,
                    'num_classes': model.backbone.fc[-1].out_features
                }, './saved_models/stamp_10_classifier.pth')
                print(f"✅ New best: {val_accuracy:.1f}%")

    return model


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📊 Val Acc: {accuracy:.1f}%")
    return accuracy

# ----------------------------------------
# 5. Testing with Similarity Matching
# ----------------------------------------


def test_stamp_matching(model, image_paths, device, tests_per_stamp=5):
    """Test using feature similarity (more realistic for production)"""
    model.eval()

    clean_transform = get_clean_transform()
    mobile_transform = get_mobile_transform()

    print("\n🎯 TESTING STAMP MATCHING:")
    print("="*60)

    # Generate reference embeddings
    reference_features = []
    with torch.no_grad():
        for path in image_paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = clean_transform(
                image=image)["image"].unsqueeze(0).to(device)

            features = model(image_tensor, return_features=True)
            reference_features.append(features)

    reference_features = torch.cat(reference_features, dim=0)

    # Test matching
    total_tests = 0
    correct_matches = 0
    per_stamp_results = []

    for stamp_idx, image_path in enumerate(image_paths):
        stamp_correct = 0

        for test_num in range(tests_per_stamp):
            # Load and heavily augment the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test_image = mobile_transform(
                image=image)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                test_features = model(test_image, return_features=True)

                # Calculate similarities with all reference features
                similarities = F.cosine_similarity(
                    test_features, reference_features, dim=1)

                # Find best match
                best_match_idx = torch.argmax(similarities).item()
                similarity_score = similarities[best_match_idx].item()

                is_correct = (best_match_idx == stamp_idx)
                if is_correct:
                    stamp_correct += 1
                    correct_matches += 1

                total_tests += 1

        stamp_accuracy = stamp_correct / tests_per_stamp * 100
        per_stamp_results.append(stamp_accuracy)

        print(
            f"Stamp {stamp_idx:2d}: {stamp_accuracy:5.1f}% ({stamp_correct}/{tests_per_stamp}) | {os.path.basename(image_path)}")

    overall_accuracy = correct_matches / total_tests * 100
    perfect_stamps = sum(1 for acc in per_stamp_results if acc == 100.0)

    print(f"\n🏆 RESULTS:")
    print(
        f"Overall Accuracy: {overall_accuracy:.1f}% ({correct_matches}/{total_tests})")
    print(f"Perfect Stamps: {perfect_stamps}/10")

    return overall_accuracy

# ----------------------------------------
# 6. Main Function
# ----------------------------------------


def main():
    # Configuration
    image_root = "./images/original"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    epochs = 40

    print(f"🔧 Device: {device}")
    print("🎯 Training classifier for 10 stamps")

    # Create datasets
    train_dataset = TenStampDataset(
        image_root, mode='train', max_images=10, samples_per_image=30)
    val_dataset = TenStampDataset(image_root, mode='val', max_images=10)

    if train_dataset.num_classes < 10:
        print(f"⚠️ Only found {train_dataset.num_classes} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"🗂️ Training samples: {len(train_dataset)}")
    print(f"🗂️ Validation samples: {len(val_dataset)}")

    # Create model
    model = Simple10StampClassifier(
        num_classes=train_dataset.num_classes).to(device)

    print(
        f"🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    model = train_model(model, train_loader, val_loader, device, epochs)

    # Test using similarity matching (more realistic)
    accuracy = test_stamp_matching(model, train_dataset.image_paths, device)

    # Results
    if accuracy >= 95:
        print("🎉 PERFECT! Ready for production!")
    elif accuracy >= 85:
        print("✅ EXCELLENT! Very good results!")
    elif accuracy >= 70:
        print("👍 GOOD! Reasonable performance!")
    else:
        print("⚠️ Needs improvement")
        print("Try: more epochs, different augmentations, or better architecture")


if __name__ == "__main__":
    main()
