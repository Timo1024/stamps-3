import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.train_dataset import StampDataset, get_stamp_augmentations
from models.encoder import StampEncoder


class StampTester:
    def __init__(self, model_path, image_root, device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.image_root = image_root

        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = StampEncoder(output_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

        # Load dataset for reference
        self.dataset = StampDataset(
            image_root=image_root,
            transform=get_stamp_augmentations()
        )

        print(f"✅ Model loaded successfully!")
        print(f"✅ Found {len(self.dataset)} reference images")
        print(f"✅ Using device: {self.device}")

        # Pre-compute embeddings for all images (optional, for faster similarity search)
        self.reference_embeddings = None
        self.reference_paths = None

    def precompute_embeddings(self):
        """Pre-compute embeddings for all images in the dataset"""
        print("🔄 Pre-computing embeddings for all images...")

        embeddings = []
        paths = []

        with torch.no_grad():
            for idx in range(len(self.dataset)):
                image, label = self.dataset[idx]
                image = image.unsqueeze(0).to(
                    self.device)  # Add batch dimension

                embedding = self.model(image)
                embeddings.append(embedding.cpu().numpy())
                paths.append(self.dataset.image_paths[idx])

                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(self.dataset)} images")

        self.reference_embeddings = np.vstack(embeddings)
        self.reference_paths = paths
        print("✅ Pre-computing completed!")

    def get_embedding(self, image_path):
        """Get embedding for a single image"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the same transforms as training
        transform = get_stamp_augmentations()
        image = transform(image=image)["image"]
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        # Get embedding
        with torch.no_grad():
            embedding = self.model(image)

        return embedding.cpu().numpy()

    def find_similar_stamps(self, query_image_path, top_k=5):
        """Find most similar stamps to a query image"""
        if self.reference_embeddings is None:
            self.precompute_embeddings()

        # Get query embedding
        query_embedding = self.get_embedding(query_image_path)

        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding, self.reference_embeddings)[0]

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'path': self.reference_paths[idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return results

    def visualize_results(self, query_image_path, results, save_path=None):
        """Visualize query image and similar results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Load and display query image
        query_img = cv2.imread(query_image_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(
            f"Query Image\n{os.path.basename(query_image_path)}")
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

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Results saved to {save_path}")

        plt.show()

    def test_embedding_quality(self, num_samples=10):
        """Test the quality of embeddings by checking if similar stamps cluster together"""
        print("🔍 Testing embedding quality...")

        if self.reference_embeddings is None:
            self.precompute_embeddings()

        # Sample some images and find their neighbors
        sample_indices = np.random.choice(
            len(self.dataset), num_samples, replace=False)

        for idx in sample_indices:
            query_path = self.reference_paths[idx]
            query_embedding = self.reference_embeddings[idx:idx+1]

            # Find similar images
            similarities = cosine_similarity(
                query_embedding, self.reference_embeddings)[0]
            top_indices = np.argsort(similarities)[
                ::-1][:6]  # Top 6 (including self)

            print(f"\nQuery: {os.path.basename(query_path)}")
            for i, sim_idx in enumerate(top_indices):
                sim_path = self.reference_paths[sim_idx]
                sim_score = similarities[sim_idx]
                marker = "🎯" if i == 0 else f"  {i}"
                print(f"{marker} {os.path.basename(sim_path)}: {sim_score:.3f}")

    def test_augmented_matching(self, num_samples=10):
        """Test if augmented images can find their original counterparts"""
        print("🔍 Testing augmented image matching...")

        if self.reference_embeddings is None:
            self.precompute_embeddings()

        # Create a transform without ToTensorV2 for getting original images
        original_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Create augmentation transform (simulating a photo taken of the stamp)
        augmented_transform = get_stamp_augmentations()

        correct_matches = 0
        total_tests = 0

        # Sample some images and test augmented versions
        sample_indices = np.random.choice(
            len(self.dataset), num_samples, replace=False)

        for idx in sample_indices:
            # Get original image path
            original_path = self.dataset.image_paths[idx]

            # Load the raw image
            image = cv2.imread(original_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create original (baseline) embedding
            original_img = original_transform(image=image)["image"]
            original_img = original_img.unsqueeze(0).to(self.device)

            # Create augmented (query) embedding
            augmented_img = augmented_transform(image=image)["image"]
            augmented_img = augmented_img.unsqueeze(0).to(self.device)

            with torch.no_grad():
                original_embedding = self.model(original_img).cpu().numpy()
                augmented_embedding = self.model(augmented_img).cpu().numpy()

            # Find most similar images to the augmented version
            similarities = cosine_similarity(
                augmented_embedding, self.reference_embeddings)[0]

            # Get top matches (excluding the exact same image if it exists)
            top_indices = np.argsort(similarities)[::-1]

            # Check if the original image is the top match
            original_idx = idx
            top_match_idx = top_indices[0]

            is_correct = (top_match_idx == original_idx)
            if is_correct:
                correct_matches += 1

            total_tests += 1

            print(f"\nTest {total_tests}: {os.path.basename(original_path)}")
            print(
                f"Original vs Augmented similarity: {cosine_similarity(original_embedding, augmented_embedding)[0][0]:.3f}")
            print(f"Top matches for augmented version:")
            for i in range(min(5, len(top_indices))):
                match_idx = top_indices[i]
                match_path = self.reference_paths[match_idx]
                match_sim = similarities[match_idx]
                marker = "✅" if match_idx == original_idx else f"  {i+1}"
                print(f"{marker} {os.path.basename(match_path)}: {match_sim:.3f}")

        accuracy = correct_matches / total_tests if total_tests > 0 else 0
        print(f"\n📊 Augmented Matching Results:")
        print(f"   Correct matches: {correct_matches}/{total_tests}")
        print(f"   Accuracy: {accuracy:.1%}")

        return accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Test the trained stamp encoder model')
    parser.add_argument('--model_path', type=str, default='./saved_models/stamp_encoder_triplet.pth',
                        help='Path to the saved model')
    parser.add_argument('--image_root', type=str, default='./images/original',
                        help='Root directory containing stamp images')
    parser.add_argument('--query_image', type=str, required=False,
                        help='Path to query image for similarity search')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of similar images to return')
    parser.add_argument('--test_quality', action='store_true',
                        help='Test embedding quality with random samples')
    parser.add_argument('--test_augmented', action='store_true',
                        help='Test if augmented images can find their original counterparts')
    parser.add_argument('--precompute', action='store_true',
                        help='Pre-compute all embeddings for faster search')

    args = parser.parse_args()

    # Initialize tester
    tester = StampTester(args.model_path, args.image_root)

    if args.precompute:
        tester.precompute_embeddings()

    if args.test_quality:
        tester.test_embedding_quality()

    if args.test_augmented:
        tester.test_augmented_matching()

    if args.query_image:
        if not os.path.exists(args.query_image):
            print(f"❌ Query image not found: {args.query_image}")
            return

        print(f"🔍 Finding similar stamps for: {args.query_image}")
        results = tester.find_similar_stamps(args.query_image, args.top_k)

        print(f"\n📊 Top {args.top_k} similar stamps:")
        for i, result in enumerate(results):
            print(
                f"{i+1}. {os.path.basename(result['path'])}: {result['similarity']:.3f}")

        # Visualize results
        save_path = f"similarity_results_{os.path.basename(args.query_image)}.png"
        tester.visualize_results(args.query_image, results, save_path)

    if not args.query_image and not args.test_quality and not args.test_augmented:
        print("ℹ️  No specific test requested. Use --help for options.")
        print("Examples:")
        print("  python test_model.py --test_quality")
        print("  python test_model.py --test_augmented")
        print("  python test_model.py --query_image path/to/image.jpg")
        print("  python test_model.py --query_image path/to/image.jpg --top_k 10")


if __name__ == "__main__":
    main()
