import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import random
from PIL import Image
import os
from pathlib import Path

# Import our custom modules
from data_loader import StampDataset
from augmentation import StampAugmentation, StampPreprocessor
from models.encoder import StampEncoder, ContrastiveLoss, TripletLoss


class StampTrainingDataset(Dataset):
    """
    PyTorch Dataset for training the stamp encoder
    """

    def __init__(self, stamp_samples, augmenter, preprocessor, samples_per_stamp=5):
        self.stamp_samples = stamp_samples
        self.augmenter = augmenter
        self.preprocessor = preprocessor
        self.samples_per_stamp = samples_per_stamp

        # Create training samples
        self.training_samples = []
        for idx, sample in enumerate(stamp_samples):
            for _ in range(samples_per_stamp):
                self.training_samples.append({
                    'stamp_data': sample,
                    'stamp_idx': idx
                })

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        stamp_data = sample['stamp_data']
        stamp_idx = sample['stamp_idx']

        # Load image
        image = Image.open(stamp_data['image_path']).convert('RGB')

        # Preprocess
        image = self.preprocessor.enhance_stamp_features(image)
        image = self.preprocessor.remove_white_background(image)

        # Create reference and augmented version
        reference = self.augmenter.process_reference(image)
        augmented = self.augmenter.simulate_user_photo(image)

        return {
            'reference': reference,
            'augmented': augmented,
            'stamp_idx': stamp_idx,
            'unique_id': stamp_data['unique_id']
        }


class StampTrainer:
    """
    Trainer class for the stamp encoder model
    """

    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        self.triplet_loss = TripletLoss(margin=0.2)

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_triplet_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            # Move to device
            reference = batch['reference'].to(self.device)
            augmented = batch['augmented'].to(self.device)
            stamp_idx = batch['stamp_idx'].to(self.device)

            # Get embeddings
            ref_embeddings = self.model(reference)
            aug_embeddings = self.model(augmented)

            # Combine embeddings and labels for contrastive loss
            all_embeddings = torch.cat([ref_embeddings, aug_embeddings], dim=0)
            all_labels = torch.cat([stamp_idx, stamp_idx], dim=0)

            # Contrastive loss
            cont_loss = self.contrastive_loss(all_embeddings, all_labels)

            # Triplet loss (reference as anchor, augmented as positive, random as negative)
            batch_size = reference.size(0)
            if batch_size > 1:
                # Create negative samples by shuffling
                neg_indices = torch.randperm(batch_size)
                negative_embeddings = ref_embeddings[neg_indices]

                # Ensure negatives are actually different stamps
                same_stamp_mask = stamp_idx == stamp_idx[neg_indices]
                if not same_stamp_mask.all():
                    trip_loss = self.triplet_loss(
                        ref_embeddings, aug_embeddings, negative_embeddings)
                else:
                    trip_loss = torch.tensor(0.0, device=self.device)
            else:
                trip_loss = torch.tensor(0.0, device=self.device)

            # Combined loss
            loss = cont_loss + 0.5 * trip_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_contrastive_loss += cont_loss.item()
            total_triplet_loss += trip_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_cont_loss = total_contrastive_loss / len(dataloader)
        avg_trip_loss = total_triplet_loss / len(dataloader)

        return avg_loss, avg_cont_loss, avg_trip_loss

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                reference = batch['reference'].to(self.device)
                augmented = batch['augmented'].to(self.device)
                stamp_idx = batch['stamp_idx'].to(self.device)

                ref_embeddings = self.model(reference)
                aug_embeddings = self.model(augmented)

                all_embeddings = torch.cat(
                    [ref_embeddings, aug_embeddings], dim=0)
                all_labels = torch.cat([stamp_idx, stamp_idx], dim=0)

                loss = self.contrastive_loss(all_embeddings, all_labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_dataloader, val_dataloader, num_epochs=20, save_path="stamp_encoder_best.pth"):
        """Full training loop"""
        best_val_loss = float('inf')

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, cont_loss, trip_loss = self.train_epoch(
                train_dataloader)

            # Validate
            val_loss = self.validate(val_dataloader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Train Loss: {train_loss:.4f} (Contrastive: {cont_loss:.4f}, Triplet: {trip_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': best_val_loss
                }, save_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")

        print(
            f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return self.train_losses, self.val_losses


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading stamp dataset...")

    # Try different possible paths for the images
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "images" / "original",
        script_dir / ".." / "images" / "original",
        Path("./images/original"),
        Path("../images/original"),
        Path("images/original")
    ]

    dataset = None
    for path in possible_paths:
        if path.exists():
            print(f"Found images at: {path}")
            dataset = StampDataset(str(path))
            break

    if dataset is None:
        print("Error: Could not find images directory!")
        print("Please ensure images are located at one of these paths:")
        for path in possible_paths:
            print(f"  {path}")
        print("Expected structure: [path]/[country]/[year]/[setID]/image.jpg")
        return

    # Get subset for fast training
    subset = dataset.get_random_subset(100)
    print(f"Using subset of {len(subset)} stamps")

    # Split into train/val
    random.shuffle(subset)
    split_idx = int(0.8 * len(subset))
    train_stamps = subset[:split_idx]
    val_stamps = subset[split_idx:]

    print(f"Train stamps: {len(train_stamps)}, Val stamps: {len(val_stamps)}")

    # Create augmentation and preprocessing
    augmenter = StampAugmentation(target_size=(224, 224))
    preprocessor = StampPreprocessor()

    # Create saved_models directory if it doesn't exist
    os.makedirs("./saved_models", exist_ok=True)

    # Create datasets
    train_dataset = StampTrainingDataset(
        train_stamps, augmenter, preprocessor, samples_per_stamp=5)
    val_dataset = StampTrainingDataset(
        val_stamps, augmenter, preprocessor, samples_per_stamp=3)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=2)

    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = StampEncoder(embedding_dim=512, backbone='resnet50')

    # Create trainer
    trainer = StampTrainer(model, device, learning_rate=1e-4)

    # Train
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=15,
        save_path="./saved_models/stamp_encoder_subset_best.pth"
    )

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_stamps': train_stamps,
        'val_stamps': val_stamps,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, "./saved_models/stamp_encoder_subset_final.pth")

    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
