from train_dataset import Ten_Classes_Dataset, get_stamp_augmentations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# Add the data directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))


class StampClassifier(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=512):
        super(StampClassifier, self).__init__()

        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)

        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original fc layer

        # Add our custom layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.feature_extractor(features)
        logits = self.classifier(embeddings)
        return embeddings, logits

    def get_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.feature_extractor(features)
        return embeddings


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings, logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            embeddings, logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def test_similarity_matching(model, reference_dataset, test_dataset, device):
    """Test matching using cosine similarity of embeddings"""
    model.eval()

    # Get reference embeddings (one per class)
    reference_images = reference_dataset.get_reference_images().to(device)
    with torch.no_grad():
        reference_embeddings = model.get_embeddings(reference_images)

    reference_embeddings_np = reference_embeddings.cpu().numpy()
    class_names = reference_dataset.get_class_names()

    print(f"Reference embeddings shape: {reference_embeddings_np.shape}")
    print(f"Class names: {class_names}")

    # Test each image in the test dataset
    correct_matches = 0
    total_tests = len(test_dataset)
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            test_image, true_label = test_dataset[idx]
            test_image = test_image.unsqueeze(0).to(device)

            # Get embedding for test image
            test_embedding = model.get_embeddings(test_image)
            test_embedding_np = test_embedding.cpu().numpy()

            # Calculate cosine similarity with all reference embeddings
            similarities = cosine_similarity(
                test_embedding_np, reference_embeddings_np)[0]

            # Find best match
            predicted_label = np.argmax(similarities)
            max_similarity = similarities[predicted_label]

            # Update statistics
            class_total[true_label] += 1
            if predicted_label == true_label:
                correct_matches += 1
                class_correct[true_label] += 1

            print(f"Test {idx}: True={class_names[true_label]}, "
                  f"Predicted={class_names[predicted_label]}, "
                  f"Similarity={max_similarity:.4f}, "
                  f"Correct={'✓' if predicted_label == true_label else '✗'}")

    # Print per-class results
    print("\n=== Per-class matching results ===")
    for class_idx in range(len(class_names)):
        if class_total[class_idx] > 0:
            accuracy = 100 * class_correct[class_idx] / class_total[class_idx]
            print(f"Class {class_idx} ({class_names[class_idx]}): "
                  f"{class_correct[class_idx]}/{class_total[class_idx]} = {accuracy:.1f}%")

    overall_accuracy = 100 * correct_matches / total_tests
    print(
        f"\n=== Overall matching accuracy: {correct_matches}/{total_tests} = {overall_accuracy:.1f}% ===")

    return overall_accuracy


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset settings
    image_root = r"e:\programming\stamps_3\testing\images\original"
    num_classes = 10
    embedding_dim = 512

    # Create datasets
    print("Creating reference dataset (no augmentation)...")
    reference_dataset = Ten_Classes_Dataset(image_root, transform=None)

    print("\nCreating training dataset (with heavy augmentation)...")
    train_dataset = Ten_Classes_Dataset(
        image_root, transform=get_stamp_augmentations())

    print("\nCreating test dataset (with augmentation to simulate mobile photos)...")
    test_dataset = Ten_Classes_Dataset(
        image_root, transform=get_stamp_augmentations())

    # Create data loaders
    # For training, we'll create multiple augmented versions of each image per epoch
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Model
    model = StampClassifier(num_classes=num_classes,
                            embedding_dim=embedding_dim)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5)

    # Training loop
    num_epochs = 50
    best_val_acc = 0

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Update learning rate
        scheduler.step(train_loss)

        # Test similarity matching every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n--- Testing similarity matching at epoch {epoch+1} ---")
            matching_acc = test_similarity_matching(
                model, reference_dataset, test_dataset, device)

            # Save best model
            if matching_acc > best_val_acc:
                best_val_acc = matching_acc
                os.makedirs('saved_models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'matching_accuracy': matching_acc,
                }, 'saved_models/best_10_class_model.pth')
                print(
                    f"New best model saved! Matching accuracy: {matching_acc:.1f}%")

    # Final test
    print(f"\n{'='*50}")
    print("FINAL SIMILARITY MATCHING TEST")
    print(f"{'='*50}")
    final_acc = test_similarity_matching(
        model, reference_dataset, test_dataset, device)

    print(f"\nFinal matching accuracy: {final_acc:.1f}%")
    print(f"Best matching accuracy during training: {best_val_acc:.1f}%")


if __name__ == "__main__":
    main()
