import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Helper for generating Triplets
# ----------------------------------------

import random


class TripletStampDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = self._create_label_indices()

    def _create_label_indices(self):
        label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getitem__(self, index):
        anchor_img, anchor_label = self.base_dataset[index]

        # Sample positive - for single image per class, we can use the same index
        # since augmentation will create different versions
        positive_candidates = [
            idx for idx in self.label_to_indices[anchor_label] if idx != index]
        if not positive_candidates:
            # If only one image per class, use the same index (augmentation will make it different)
            positive_index = index
        else:
            positive_index = random.choice(positive_candidates)
        positive_img, _ = self.base_dataset[positive_index]

        # Sample negative
        negative_labels = [
            label for label in self.label_to_indices.keys() if label != anchor_label]
        if not negative_labels:
            # Fallback if only one class exists (shouldn't happen in practice)
            raise ValueError("Cannot create triplets with only one class")
        else:
            negative_label = random.choice(negative_labels)
            negative_index = random.choice(
                self.label_to_indices[negative_label])
        negative_img, _ = self.base_dataset[negative_index]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)

# ----------------------------------------
# 2. Training Function
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
# 3. Setup + Run
# ----------------------------------------


if __name__ == "__main__":
    image_root = "./images/original"
    batch_size = 8  # 32 -> 8 faster?
    embedding_dim = 128
    epochs = 10  # 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")

    # Load dataset
    base_dataset = StampDataset(
        image_root=image_root,
        transform=get_stamp_augmentations()
    )

    print(f"✅ Found {len(base_dataset)} images")

    triplet_dataset = TripletStampDataset(base_dataset)

    # Check dataset integrity
    print(
        f"✅ Number of unique labels: {len(triplet_dataset.label_to_indices)}")
    print(
        f"✅ Images per label: {len(base_dataset) // len(triplet_dataset.label_to_indices)}")

    train_loader = DataLoader(
        triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Model, loss, optimizer
    model = StampEncoder(output_dim=embedding_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Test a single batch to debug tensor types
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
    torch.save(model.state_dict(), "./saved_models/stamp_encoder_triplet.pth")
    print("✅ Model saved to ./saved_models/stamp_encoder_triplet.pth")
