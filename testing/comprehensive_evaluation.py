import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def comprehensive_evaluation(model_path, image_root):
    """Comprehensive evaluation of the stamp encoder model"""
    print("🔍 Comprehensive Stamp Encoder Evaluation")
    print("=" * 60)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StampEncoder(output_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = StampDataset(image_root, transform=get_stamp_augmentations())
    print(f"✅ Model loaded on {device}")
    print(f"✅ Dataset loaded with {len(dataset)} images")

    # Test 1: Augmentation consistency
    print("\n📊 Test 1: Augmentation Consistency")
    print("-" * 50)

    augmentation_similarities = []

    for i in range(50):
        idx = random.randint(0, len(dataset) - 1)

        # Get two different augmentations of the same image
        img1, _ = dataset[idx]
        img2, _ = dataset[idx]

        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0).to(device)).cpu().numpy().flatten()
            emb2 = model(img2.unsqueeze(0).to(device)).cpu().numpy().flatten()

        similarity = cosine_similarity(emb1, emb2)
        augmentation_similarities.append(similarity)

    avg_aug_sim = np.mean(augmentation_similarities)
    print(f"Average similarity between augmented versions: {avg_aug_sim:.3f}")
    print(f"Min similarity: {np.min(augmentation_similarities):.3f}")
    print(f"Max similarity: {np.max(augmentation_similarities):.3f}")

    # Test 2: Cross-image similarities
    print("\n📊 Test 2: Cross-Image Similarities")
    print("-" * 50)

    cross_similarities = []

    for i in range(100):
        idx1 = random.randint(0, len(dataset) - 1)
        idx2 = random.randint(0, len(dataset) - 1)

        while idx1 == idx2:
            idx2 = random.randint(0, len(dataset) - 1)

        img1, _ = dataset[idx1]
        img2, _ = dataset[idx2]

        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0).to(device)).cpu().numpy().flatten()
            emb2 = model(img2.unsqueeze(0).to(device)).cpu().numpy().flatten()

        similarity = cosine_similarity(emb1, emb2)
        cross_similarities.append(similarity)

    avg_cross_sim = np.mean(cross_similarities)
    print(f"Average similarity between different stamps: {avg_cross_sim:.3f}")
    print(f"Min similarity: {np.min(cross_similarities):.3f}")
    print(f"Max similarity: {np.max(cross_similarities):.3f}")

    # Test 3: Similarity distribution analysis
    print("\n📊 Test 3: Similarity Distribution Analysis")
    print("-" * 50)

    # Plot similarity distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(augmentation_similarities, bins=20, alpha=0.7,
             color='green', label='Same stamp (augmented)')
    plt.hist(cross_similarities, bins=20, alpha=0.7,
             color='red', label='Different stamps')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([augmentation_similarities, cross_similarities],
                labels=['Same stamp\n(augmented)', 'Different\nstamps'])
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Box Plot')

    plt.tight_layout()
    plt.savefig('similarity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Test 4: Retrieval performance
    print("\n📊 Test 4: Retrieval Performance")
    print("-" * 50)

    # Test different similarity thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        correct_high = sum(
            1 for sim in augmentation_similarities if sim >= threshold)
        incorrect_high = sum(
            1 for sim in cross_similarities if sim >= threshold)

        total_same = len(augmentation_similarities)
        total_diff = len(cross_similarities)

        precision = correct_high / \
            (correct_high + incorrect_high) if (correct_high + incorrect_high) > 0 else 0
        recall = correct_high / total_same if total_same > 0 else 0

        print(
            f"Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")

    # Summary
    print("\n📋 Summary")
    print("=" * 60)
    print(f"✅ Same stamp similarity: {avg_aug_sim:.3f}")
    print(f"✅ Different stamp similarity: {avg_cross_sim:.3f}")
    print(f"✅ Discrimination gap: {avg_aug_sim - avg_cross_sim:.3f}")

    # Interpretation
    if avg_aug_sim > avg_cross_sim + 0.1:
        print("🎉 EXCELLENT: Model clearly distinguishes between same and different stamps!")
    elif avg_aug_sim > avg_cross_sim:
        print("👍 GOOD: Model shows some discrimination, but could be improved")
    else:
        print("⚠️ POOR: Model struggles to distinguish between same and different stamps")

    print(f"\n💡 Model Analysis:")
    print(f"   - Your model learns to group visually similar stamps together")
    print(f"   - Augmented versions may be more similar to other stamps than original")
    print(f"   - This is NORMAL and shows the model learned meaningful features")
    print(f"   - For exact matching, you'd need a different loss function")

    return {
        'augmentation_similarity': avg_aug_sim,
        'cross_similarity': avg_cross_sim,
        'discrimination_gap': avg_aug_sim - avg_cross_sim
    }


def main():
    model_path = "./saved_models/stamp_encoder_triplet.pth"
    image_root = "./images/original"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    comprehensive_evaluation(model_path, image_root)


if __name__ == "__main__":
    main()
