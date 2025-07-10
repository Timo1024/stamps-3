import torch
import numpy as np
import os
import random
from collections import defaultdict

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_model(model_path, image_root, device='cuda'):
    """Evaluate the model by testing embedding quality"""

    print("🔍 Model Evaluation")
    print("=" * 50)

    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = StampEncoder(output_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = StampDataset(image_root, transform=get_stamp_augmentations())

    print(f"✅ Model loaded on {device}")
    print(f"✅ Dataset loaded with {len(dataset)} images")

    # Test 1: Same image with different augmentations should be similar
    print("\n📊 Test 1: Consistency with augmentations")
    print("-" * 40)

    same_image_similarities = []

    for i in range(10):  # Test 10 random images
        idx = random.randint(0, len(dataset) - 1)

        # Get two different augmentations of the same image
        img1, label1 = dataset[idx]
        img2, label2 = dataset[idx]

        # Get embeddings
        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0).to(device)).cpu().numpy().flatten()
            emb2 = model(img2.unsqueeze(0).to(device)).cpu().numpy().flatten()

        similarity = cosine_similarity(emb1, emb2)
        same_image_similarities.append(similarity)

        print(f"Image {i+1} self-similarity: {similarity:.3f}")

    avg_self_similarity = np.mean(same_image_similarities)
    print(f"Average self-similarity: {avg_self_similarity:.3f}")

    # Test 2: Different images should be less similar than same image
    print("\n📊 Test 2: Discrimination between different images")
    print("-" * 40)

    different_image_similarities = []

    for i in range(20):  # Test 20 pairs
        idx1 = random.randint(0, len(dataset) - 1)
        idx2 = random.randint(0, len(dataset) - 1)

        # Make sure they're different images
        while idx1 == idx2:
            idx2 = random.randint(0, len(dataset) - 1)

        # Get embeddings
        img1, _ = dataset[idx1]
        img2, _ = dataset[idx2]

        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0).to(device)).cpu().numpy().flatten()
            emb2 = model(img2.unsqueeze(0).to(device)).cpu().numpy().flatten()

        similarity = cosine_similarity(emb1, emb2)
        different_image_similarities.append(similarity)

    avg_different_similarity = np.mean(different_image_similarities)
    print(
        f"Average different-image similarity: {avg_different_similarity:.3f}")

    # Test 3: Embedding distribution analysis
    print("\n📊 Test 3: Embedding distribution")
    print("-" * 40)

    # Sample embeddings
    sample_embeddings = []
    for i in range(100):
        idx = random.randint(0, len(dataset) - 1)
        img, _ = dataset[idx]

        with torch.no_grad():
            emb = model(img.unsqueeze(0).to(device)).cpu().numpy().flatten()
        sample_embeddings.append(emb)

    sample_embeddings = np.array(sample_embeddings)

    # Calculate statistics
    mean_norm = np.mean(np.linalg.norm(sample_embeddings, axis=1))
    std_norm = np.std(np.linalg.norm(sample_embeddings, axis=1))

    print(f"Average embedding norm: {mean_norm:.3f} ± {std_norm:.3f}")
    print(f"Embedding dimension: {sample_embeddings.shape[1]}")

    # Summary
    print("\n📋 Summary")
    print("=" * 50)
    print(f"✅ Self-similarity (same image): {avg_self_similarity:.3f}")
    print(
        f"✅ Cross-similarity (different images): {avg_different_similarity:.3f}")
    print(
        f"✅ Discrimination gap: {avg_self_similarity - avg_different_similarity:.3f}")

    if avg_self_similarity > avg_different_similarity:
        print("🎉 Model shows good discrimination!")
    else:
        print("⚠️  Model may need more training - self-similarity should be higher than cross-similarity")

    return {
        'self_similarity': avg_self_similarity,
        'cross_similarity': avg_different_similarity,
        'discrimination_gap': avg_self_similarity - avg_different_similarity
    }


def main():
    model_path = "./saved_models/stamp_encoder_triplet.pth"
    image_root = "./images/original"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return

    evaluate_model(model_path, image_root)


if __name__ == "__main__":
    main()
