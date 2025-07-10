import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder


def load_model(model_path, device='cuda'):
    """Load the trained model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = StampEncoder(output_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def get_embedding(model, image_path, device):
    """Get embedding for a single image"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the same transforms as training
    transform = get_stamp_augmentations()
    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    # Get embedding
    with torch.no_grad():
        embedding = model(image)

    return embedding.cpu().numpy()


def find_similar_images(model, query_image_path, reference_paths, device, top_k=5):
    """Find similar images to a query"""
    query_embedding = get_embedding(model, query_image_path, device)

    # Get embeddings for all reference images
    reference_embeddings = []
    valid_paths = []

    print(f"Processing {len(reference_paths)} reference images...")
    for i, ref_path in enumerate(reference_paths):
        try:
            ref_embedding = get_embedding(model, ref_path, device)
            reference_embeddings.append(ref_embedding)
            valid_paths.append(ref_path)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(reference_paths)} images")
        except Exception as e:
            print(f"Skipping {ref_path}: {e}")

    reference_embeddings = np.vstack(reference_embeddings)

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, reference_embeddings)[0]

    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'path': valid_paths[idx],
            'similarity': similarities[idx]
        })

    return results


def visualize_similarity_results(query_image_path, results):
    """Visualize the similarity search results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Load and display query image
    query_img = cv2.imread(query_image_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f"Query Image\n{os.path.basename(query_image_path)}")
    axes[0, 0].axis('off')

    # Display top 5 similar images
    for i, result in enumerate(results[:5]):
        row = (i + 1) // 3
        col = (i + 1) % 3

        img = cv2.imread(result['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f"Similarity: {result['similarity']:.3f}\n{os.path.basename(result['path'])}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def test_random_samples(model, image_paths, device, num_tests=5):
    """Test with random samples from the dataset"""
    print(f"🎲 Testing with {num_tests} random samples...")

    for i in range(num_tests):
        # Pick a random query image
        query_path = random.choice(image_paths)

        # Find similar images
        other_paths = [p for p in image_paths if p != query_path]
        results = find_similar_images(
            model, query_path, other_paths[:200], device, top_k=5)

        print(f"\n--- Test {i+1} ---")
        print(f"Query: {os.path.basename(query_path)}")
        print("Most similar images:")
        for j, result in enumerate(results):
            print(
                f"  {j+1}. {os.path.basename(result['path'])}: {result['similarity']:.3f}")

        # Visualize first test
        if i == 0:
            visualize_similarity_results(query_path, results)


def main():
    print("🎯 Stamp Encoder Model Tester")
    print("=" * 50)

    # Configuration
    model_path = "./saved_models/stamp_encoder_triplet.pth"
    image_root = "./images/original"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return

    # Load model
    print("Loading model...")
    model, device = load_model(model_path)
    print(f"✅ Model loaded successfully on {device}")

    # Load dataset
    # No transform for testing
    dataset = StampDataset(image_root, transform=None)
    image_paths = dataset.image_paths
    print(f"✅ Found {len(image_paths)} images")

    while True:
        print("\n" + "=" * 50)
        print("Choose an option:")
        print("1. Test with a specific image")
        print("2. Test with random samples")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            # Test with specific image
            query_path = input("Enter path to query image: ").strip()

            if not os.path.exists(query_path):
                print(f"❌ Image not found: {query_path}")
                continue

            try:
                top_k = int(
                    input("Number of similar images to find (default 5): ") or "5")

                print(
                    f"🔍 Finding similar stamps for: {os.path.basename(query_path)}")
                results = find_similar_images(
                    model, query_path, image_paths, device, top_k)

                print(f"\n📊 Top {top_k} similar stamps:")
                for i, result in enumerate(results):
                    print(
                        f"{i+1}. {os.path.basename(result['path'])}: {result['similarity']:.3f}")

                # Visualize results
                visualize_similarity_results(query_path, results)

            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "2":
            # Test with random samples
            try:
                num_tests = int(
                    input("Number of random tests (default 3): ") or "3")
                test_random_samples(model, image_paths, device, num_tests)
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
