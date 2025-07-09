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

        # Sample positive
        positive_index = random.choice(self.label_to_indices[anchor_label])
        while positive_index == index:
            positive_index = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self.base_dataset[positive_index]

        # Sample negative
        negative_label = random.choice(list(self.label_to_indices.keys()))
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.label_to_indices.keys()))
        negative_index = random.choice(self.label_to_indices[negative_label])
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

            print("Processing batch...")

            anchor = anchor.to(device)

            print(f"Anchor shape: {anchor.shape}")

            positive = positive.to(device)

            print(f"Positive shape: {positive.shape}")

            negative = negative.to(device)

            print(f"Negative shape: {negative.shape}")

            optimizer.zero_grad()

            print("Forward pass...")

            anchor_out = model(anchor)

            print(f"Anchor output shape: {anchor_out.shape}")

            positive_out = model(positive)

            print(f"Positive output shape: {positive_out.shape}")

            negative_out = model(negative)

            print(f"Negative output shape: {negative_out.shape}")

            loss = criterion(anchor_out, positive_out, negative_out)

            print(f"Loss: {loss.item()}")

            loss.backward()

            print("Backward pass...")

            optimizer.step()

            print("Optimizer step...")

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader)}")

# ----------------------------------------
# 3. Setup + Run
# ----------------------------------------


if __name__ == "__main__":
    image_root = "./images/original"
    batch_size = 8  # 32
    embedding_dim = 128
    epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")

    # Load dataset
    base_dataset = StampDataset(
        image_root=image_root,
        transform=get_stamp_augmentations()
    )
    triplet_dataset = TripletStampDataset(base_dataset)
    train_loader = DataLoader(
        triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Model, loss, optimizer
    model = StampEncoder(output_dim=embedding_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train(model, train_loader, criterion, optimizer, device, epochs=epochs)

    # Save model
    os.makedirs("./saved_models", exist_ok=True)
    torch.save(model.state_dict(), "./saved_models/stamp_encoder_triplet.pth")
    print("✅ Model saved to ./saved_models/stamp_encoder_triplet.pth")
