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

    return embedding.cpu().numpy().flatten()


def find_similar_images(model, query_image_path, reference_paths, device, top_k=5):
    """Find similar images to a query"""
    query_embedding = get_embedding(model, query_image_path, device)

    similarities = []
    valid_paths = []

    print(f"Processing {len(reference_paths)} reference images...")
    for i, ref_path in enumerate(reference_paths):
        try:
            ref_embedding = get_embedding(model, ref_path, device)
            similarity = cosine_similarity(query_embedding, ref_embedding)
            similarities.append(similarity)
            valid_paths.append(ref_path)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(reference_paths)} images")
        except Exception as e:
            print(f"Skipping {ref_path}: {e}")

    # Get top-k most similar
    similarities = np.array(similarities)
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


def quick_test_model(model, image_paths, device):
    """Quick test with a few random images"""
    print("🎯 Quick Model Test")
    print("-" * 30)

    # Pick 3 random images
    test_images = random.sample(image_paths, min(3, len(image_paths)))

    for i, query_path in enumerate(test_images):
        print(f"\n--- Test {i+1} ---")
        print(f"Query: {os.path.basename(query_path)}")

        # Find similar images (sample 100 random images for faster testing)
        sample_paths = random.sample(
            [p for p in image_paths if p != query_path], min(100, len(image_paths)-1))

        results = find_similar_images(
            model, query_path, sample_paths, device, top_k=3)

        print("Most similar images:")
        for j, result in enumerate(results):
            print(
                f"  {j+1}. {os.path.basename(result['path'])}: {result['similarity']:.3f}")

        # Show visualization for first test
        if i == 0:
            print("Showing visualization for first test...")
            visualize_similarity_results(query_path, results)


def interactive_test(model, image_paths, device):
    """Interactive testing interface"""
    while True:
        print("\n" + "=" * 50)
        print("🔍 Interactive Stamp Similarity Search")
        print("=" * 50)
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

                # Use a subset for faster testing
                sample_size = min(500, len(image_paths))
                sample_paths = random.sample(image_paths, sample_size)

                results = find_similar_images(
                    model, query_path, sample_paths, device, top_k)

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

                for i in range(num_tests):
                    query_path = random.choice(image_paths)
                    sample_paths = random.sample(
                        [p for p in image_paths if p != query_path], min(200, len(image_paths)-1))

                    results = find_similar_images(
                        model, query_path, sample_paths, device, top_k=3)

                    print(f"\n--- Random Test {i+1} ---")
                    print(f"Query: {os.path.basename(query_path)}")
                    print("Most similar images:")
                    for j, result in enumerate(results):
                        print(
                            f"  {j+1}. {os.path.basename(result['path'])}: {result['similarity']:.3f}")

                    if i == 0:
                        visualize_similarity_results(query_path, results)

            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def test_augmented_matching(model, image_paths, device, num_tests=10):
    """Test if augmented images can find their original counterparts"""
    print("🔍 Testing Augmented Image Matching")
    print("-" * 40)

    # Create transforms
    from data.train_dataset import get_stamp_augmentations
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Original transform (minimal processing)
    original_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Augmented transform (simulating phone camera)
    augmented_transform = get_stamp_augmentations()

    correct_matches = 0
    total_tests = 0

    print(f"Testing {num_tests} random stamps...")

    for i in range(num_tests):
        # Pick a random image
        original_path = random.choice(image_paths)

        # Load raw image
        image = cv2.imread(original_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create original and augmented versions
        original_img = original_transform(image=image)["image"]
        augmented_img = augmented_transform(image=image)["image"]

        # Get embeddings
        original_embedding = get_embedding(model, original_path, device)

        with torch.no_grad():
            augmented_embedding = model(augmented_img.unsqueeze(
                0).to(device)).cpu().numpy().flatten()

        # Test similarity between original and augmented
        self_similarity = cosine_similarity(original_embedding.reshape(1, -1),
                                            augmented_embedding.reshape(1, -1))[0][0]

        # Find most similar images to augmented version
        other_paths = [p for p in image_paths if p != original_path]
        sample_paths = random.sample(other_paths, min(100, len(other_paths)))

        best_match_similarity = 0
        best_match_path = None

        for other_path in sample_paths:
            try:
                other_embedding = get_embedding(model, other_path, device)
                similarity = cosine_similarity(augmented_embedding.reshape(1, -1),
                                               other_embedding.reshape(1, -1))[0][0]
                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_path = other_path
            except:
                continue

        # Check if original is more similar than best alternative
        is_correct = self_similarity > best_match_similarity
        if is_correct:
            correct_matches += 1

        total_tests += 1

        print(f"\nTest {total_tests}: {os.path.basename(original_path)}")
        print(f"  Original ↔ Augmented: {self_similarity:.3f}")
        print(
            f"  Best other match: {best_match_similarity:.3f} ({os.path.basename(best_match_path) if best_match_path else 'None'})")
        print(f"  Result: {'✅ CORRECT' if is_correct else '❌ WRONG'}")

    accuracy = correct_matches / total_tests if total_tests > 0 else 0
    print(f"\n📊 Augmented Matching Results:")
    print(f"   Correct matches: {correct_matches}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1%}")

    if accuracy > 0.8:
        print("🎉 Excellent! Model can reliably match augmented images!")
    elif accuracy > 0.6:
        print("👍 Good performance, but could be improved")
    else:
        print("⚠️ Poor performance - model may need more training")

    return accuracy


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
    # No transform for listing
    dataset = StampDataset(image_root, transform=None)
    image_paths = dataset.image_paths
    # Ask user what they want to do
    print(f"✅ Found {len(image_paths)} images")
    print("\nWhat would you like to do?")
    print("1. Quick test (automatic with random images)")
    print("2. Interactive test (choose your own images)")
    print("3. Test augmented matching (simulates phone photos)")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        quick_test_model(model, image_paths, device)
    elif choice == "2":
        interactive_test(model, image_paths, device)
    elif choice == "3":
        test_augmented_matching(model, image_paths, device)
    else:
        print("❌ Invalid choice. Running quick test...")
        quick_test_model(model, image_paths, device)


if __name__ == "__main__":
    main()
