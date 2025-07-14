# Simple Stamp Matcher without FAISS dependency
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple, Dict
import time
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import StampDataset
from augmentation import StampAugmentation, StampPreprocessor
from models.encoder import StampEncoder


class SimpleStampMatcher:
    """
    Simple stamp matcher using cosine similarity (no FAISS dependency)
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.embeddings = None
        self.stamp_database = []
        self.augmenter = StampAugmentation(target_size=(224, 224))
        self.preprocessor = StampPreprocessor()

        # Load model
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained encoder model"""
        print(f"Loading model from {model_path}...")

        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            print("Please train the model first by running train_encoder.py")
            return

        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model
        self.model = StampEncoder(embedding_dim=512, backbone='resnet50')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def create_database_embeddings(self, stamp_samples: List[Dict]):
        """Create embeddings for all stamps in the database"""
        print(f"Creating embeddings for {len(stamp_samples)} stamps...")

        embeddings = []
        self.stamp_database = []

        with torch.no_grad():
            for i, sample in enumerate(stamp_samples):
                try:
                    # Load and preprocess image
                    image = Image.open(sample['image_path']).convert('RGB')
                    image = self.preprocessor.enhance_stamp_features(image)
                    image = self.preprocessor.remove_white_background(image)

                    # Process as reference image
                    img_tensor = self.augmenter.process_reference(
                        image).unsqueeze(0).to(self.device)

                    # Get embedding
                    embedding = self.model(img_tensor).cpu().numpy()
                    embeddings.append(embedding[0])
                    self.stamp_database.append(sample)

                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(stamp_samples)} stamps")

                except Exception as e:
                    print(f"Error processing {sample['image_path']}: {e}")
                    continue

        # Convert to numpy array and normalize
        self.embeddings = np.array(embeddings)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

        print(f"Database created with {len(self.stamp_database)} stamps")
        return self.embeddings

    def find_matches(self, user_image: Image.Image, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Find the top-k most similar stamps to the user's image
        """
        if self.model is None or self.embeddings is None:
            raise ValueError("Model and database must be loaded first")

        start_time = time.time()

        # Preprocess user image
        processed_image = self.preprocessor.enhance_stamp_features(user_image)
        processed_image = self.preprocessor.remove_white_background(
            processed_image)

        # Convert to tensor and get embedding
        img_tensor = self.augmenter.simulate_user_photo(
            processed_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            user_embedding = self.model(img_tensor).cpu().numpy()

        # Normalize user embedding
        user_embedding = user_embedding / \
            (np.linalg.norm(user_embedding) + 1e-8)

        # Calculate cosine similarities
        similarities = cosine_similarity(user_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Prepare results
        matches = []
        for idx in top_indices:
            matches.append(
                (self.stamp_database[idx], float(similarities[idx])))

        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.3f} seconds")

        return matches

    def evaluate_on_test_set(self, test_stamps: List[Dict], top_k_values: List[int] = [1, 5, 10]):
        """
        Evaluate the matcher on a test set
        """
        print(f"Evaluating on {len(test_stamps)} test stamps...")

        results = {k: 0 for k in top_k_values}
        total_tests = 0

        for i, test_stamp in enumerate(test_stamps):
            try:
                # Load test image
                image = Image.open(test_stamp['image_path']).convert('RGB')

                # Find matches
                matches = self.find_matches(image, top_k=max(top_k_values))

                # Check if correct stamp is in top-k
                test_id = test_stamp['unique_id']

                for k in top_k_values:
                    top_k_matches = matches[:k]
                    if any(match[0]['unique_id'] == test_id for match in top_k_matches):
                        results[k] += 1

                total_tests += 1

                if (i + 1) % 5 == 0:
                    print(f"Evaluated {i + 1}/{len(test_stamps)} stamps")

            except Exception as e:
                print(f"Error evaluating {test_stamp['image_path']}: {e}")
                continue

        # Calculate accuracy
        print("\nEvaluation Results:")
        for k in top_k_values:
            accuracy = results[k] / total_tests * 100
            print(
                f"Top-{k} Accuracy: {accuracy:.2f}% ({results[k]}/{total_tests})")

        return results


def main():
    """Demo of the simple stamp matching system"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")

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
        return
    subset = dataset.get_random_subset(100)

    # Split for testing
    train_size = int(0.8 * len(subset))
    train_stamps = subset[:train_size]
    test_stamps = subset[train_size:]

    print(
        f"Using {len(train_stamps)} stamps for database, {len(test_stamps)} for testing")

    # Create matcher
    model_path = "./saved_models/stamp_encoder_subset_best.pth"
    matcher = SimpleStampMatcher(model_path, device=device)

    if matcher.model is None:
        print("Model not loaded. Please train the model first.")
        return

    # Create database embeddings
    embeddings = matcher.create_database_embeddings(train_stamps)

    # Evaluate on test set
    if test_stamps:
        results = matcher.evaluate_on_test_set(
            test_stamps, top_k_values=[1, 5, 10])

        # Test with a specific image
        print(f"\nTesting with image: {test_stamps[0]['unique_id']}")
        test_image = Image.open(test_stamps[0]['image_path']).convert('RGB')

        matches = matcher.find_matches(test_image, top_k=5)

        print("Top 5 matches:")
        for i, (match, similarity) in enumerate(matches):
            print(
                f"{i+1}. {match['unique_id']} (similarity: {similarity:.4f})")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
