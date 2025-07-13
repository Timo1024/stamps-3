import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
from tqdm import tqdm

from data.train_dataset import StampDataset
from models.encoder import StampEncoder


def get_simple_test_augmentations():
    """Simple augmentations for debugging"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_reference_augmentations():
    """Clean reference augmentations"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def debug_model_embeddings():
    """Debug model embeddings to see if they're discriminative"""
    # Configuration
    model_path = "./saved_models/stamp_encoder_advanced.pth"
    image_root = "./images/original"
    embedding_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Using device: {device}")
    print(f"📁 Loading model from: {model_path}")

    # Load model
    model = StampEncoder(output_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load small sample of dataset
    base_dataset = StampDataset(image_root=image_root, transform=None)
    test_transform = get_simple_test_augmentations()
    ref_transform = get_reference_augmentations()

    print(f"📊 Testing with {len(base_dataset)} total images")

    # Test with first 10 images
    test_indices = list(range(min(10, len(base_dataset))))

    print("\\n🔍 DEBUGGING EMBEDDINGS:")
    print("=" * 50)

    with torch.no_grad():
        embeddings_ref = []
        embeddings_aug = []

        for idx in test_indices:
            # Load raw image
            image_path = base_dataset.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reference embedding
            ref_img = ref_transform(image=image)[
                "image"].unsqueeze(0).to(device)
            ref_embedding = model(ref_img)
            ref_embedding = F.normalize(
                ref_embedding, p=2, dim=1).cpu().squeeze()

            # Augmented embedding
            aug_img = test_transform(image=image)[
                "image"].unsqueeze(0).to(device)
            aug_embedding = model(aug_img)
            aug_embedding = F.normalize(
                aug_embedding, p=2, dim=1).cpu().squeeze()

            embeddings_ref.append(ref_embedding)
            embeddings_aug.append(aug_embedding)

            print(f"Image {idx}:")
            print(f"  Ref embedding norm: {torch.norm(ref_embedding):.4f}")
            print(f"  Aug embedding norm: {torch.norm(aug_embedding):.4f}")
            self_sim = F.cosine_similarity(ref_embedding.unsqueeze(
                0), aug_embedding.unsqueeze(0)).item()
            print(f"  Self similarity: {self_sim:.4f}")

    print("\\n🔍 CROSS-IMAGE SIMILARITIES:")
    print("=" * 50)

    # Calculate similarities between different images
    similarity_matrix = torch.zeros(len(test_indices), len(test_indices))

    for i in range(len(test_indices)):
        for j in range(len(test_indices)):
            sim = F.cosine_similarity(
                embeddings_ref[i].unsqueeze(0),
                embeddings_ref[j].unsqueeze(0)
            ).item()
            similarity_matrix[i, j] = sim

    print("Similarity matrix (reference embeddings):")
    print(similarity_matrix)

    # Statistics
    diag_similarities = torch.diag(similarity_matrix)
    off_diag_mask = ~torch.eye(len(test_indices), dtype=bool)
    off_diag_similarities = similarity_matrix[off_diag_mask]

    print(f"\\nDiagonal (self) similarities: {diag_similarities}")
    print(
        f"Off-diagonal similarities - Mean: {off_diag_similarities.mean():.4f}")
    print(
        f"Off-diagonal similarities - Std: {off_diag_similarities.std():.4f}")
    print(
        f"Off-diagonal similarities - Min: {off_diag_similarities.min():.4f}")
    print(
        f"Off-diagonal similarities - Max: {off_diag_similarities.max():.4f}")

    if off_diag_similarities.mean() > 0.9:
        print("⚠️  WARNING: Very high inter-image similarity detected!")
        print("   This suggests the model is not learning discriminative features.")
    elif off_diag_similarities.mean() > 0.8:
        print("⚠️  CAUTION: High inter-image similarity detected.")
    else:
        print("✅ Good inter-image similarity separation.")

    # Test actual matching
    print("\\n🎯 TESTING MATCHING:")
    print("=" * 50)

    correct_matches = 0
    total_tests = 0

    for query_idx in range(len(test_indices)):
        query_embedding = embeddings_aug[query_idx]

        # Calculate similarities with all reference images
        similarities = []
        for ref_idx in range(len(test_indices)):
            ref_embedding = embeddings_ref[ref_idx]
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                ref_embedding.unsqueeze(0)
            ).item()
            similarities.append((sim, ref_idx))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        predicted_idx = similarities[0][1]

        is_correct = (predicted_idx == query_idx)
        correct_matches += is_correct
        total_tests += 1

        print(
            f"Query {query_idx}: Predicted {predicted_idx}, Correct: {is_correct}")
        print(f"  Top 3 matches: {similarities[:3]}")

    accuracy = correct_matches / total_tests if total_tests > 0 else 0
    print(
        f"\\n🎯 Accuracy on small test: {accuracy*100:.1f}% ({correct_matches}/{total_tests})")


if __name__ == "__main__":
    debug_model_embeddings()
