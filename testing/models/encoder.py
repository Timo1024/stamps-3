import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class StampEncoder(nn.Module):
    """
    CNN-based encoder for extracting features from stamp images
    """

    def __init__(self, embedding_dim=512, backbone='resnet50'):
        super(StampEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classification layer
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            backbone_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)

        # Project to embedding space
        embeddings = self.projection_head(features)

        # Normalize embeddings
        embeddings = self.norm(embeddings)

        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similar representations for the same stamp
    and different representations for different stamps
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, embedding_dim]
        labels: [batch_size] - stamp IDs
        """
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(
            embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same stamp)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)

        # Sum of all similarities for normalization
        sum_exp_sim = torch.sum(
            exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1, keepdim=True)

        # Positive pairs
        pos_sim = torch.sum(exp_sim * mask, dim=1, keepdim=True)

        # Avoid division by zero
        pos_sim = torch.clamp(pos_sim, min=1e-8)
        sum_exp_sim = torch.clamp(sum_exp_sim, min=1e-8)

        # Contrastive loss
        loss = -torch.log(pos_sim / sum_exp_sim)

        # Only consider samples that have positive pairs
        mask_pos = torch.sum(mask, dim=1) > 0
        if torch.sum(mask_pos) > 0:
            loss = torch.mean(loss[mask_pos])
        else:
            loss = torch.tensor(
                0.0, device=embeddings.device, requires_grad=True)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss: anchor-positive distance should be smaller than anchor-negative distance
    """

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: [batch_size, embedding_dim]
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, 2)
        neg_dist = F.pairwise_distance(anchor, negative, 2)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(loss)


class SimpleClassifier(nn.Module):
    """
    Simple classifier for stamp recognition (for comparison)
    """

    def __init__(self, num_classes, backbone='resnet50'):
        super(SimpleClassifier, self).__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(backbone_dim, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test StampEncoder
    encoder = StampEncoder(embedding_dim=512).to(device)

    # Create dummy input
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        embeddings = encoder(dummy_input)
        print(f"Encoder output shape: {embeddings.shape}")
        print(
            f"Embedding norm (should be ~1.0): {torch.norm(embeddings, dim=1).mean().item():.4f}")

    # Test ContrastiveLoss
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(
        device)  # 4 different stamps, 2 samples each
    contrastive_loss = ContrastiveLoss()

    loss = contrastive_loss(embeddings, labels)
    print(f"Contrastive loss: {loss.item():.4f}")

    # Test TripletLoss
    anchor = embeddings[:4]  # First 4 embeddings
    positive = embeddings[:4]  # Same stamps (for testing)
    negative = embeddings[4:]  # Different stamps

    triplet_loss = TripletLoss()
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet loss: {loss.item():.4f}")

    print("Model tests completed successfully!")
