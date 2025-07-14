import torch
import numpy as np
from PIL import Image
import faiss
import json
from pathlib import Path
from typing import List, Tuple, Dict
import time

from data_loader import StampDataset
from augmentation import StampAugmentation, StampPreprocessor
from models.encoder import StampEncoder


class StampMatcher:
    """
    Class for matching user-uploaded stamp images to the database
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.index = None
        self.stamp_database = []
        self.augmenter = StampAugmentation(target_size=(224, 224))
        self.preprocessor = StampPreprocessor()

        # Load model
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained encoder model"""
        print(f"Loading model from {model_path}...")
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

                    if (i + 1) % 50 == 0:
                        print(f"Processed {i + 1}/{len(stamp_samples)} stamps")

                except Exception as e:
                    print(f"Error processing {sample['image_path']}: {e}")
                    continue

        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index for fast similarity search
        # Inner product (cosine similarity)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)

        print(f"Database created with {len(self.stamp_database)} stamps")
        return embeddings

    def find_matches(self, user_image: Image.Image, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Find the top-k most similar stamps to the user's image
        """
        if self.model is None or self.index is None:
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
            user_embedding = self.model(
                img_tensor).cpu().numpy().astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(user_embedding)

        # Search in the database
        similarities, indices = self.index.search(user_embedding, top_k)

        # Prepare results
        matches = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.stamp_database):
                matches.append((self.stamp_database[idx], float(similarity)))

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

                if (i + 1) % 10 == 0:
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

    def save_database(self, filepath: str):
        """Save the database and index"""
        if self.index is None:
            raise ValueError("Database must be created first")

        # Save FAISS index
        faiss.write_index(self.index, filepath + ".faiss")

        # Save stamp database
        with open(filepath + ".json", 'w') as f:
            json.dump(self.stamp_database, f, indent=2)

        print(f"Database saved to {filepath}.faiss and {filepath}.json")

    def load_database(self, filepath: str):
        """Load saved database and index"""
        # Load FAISS index
        self.index = faiss.read_index(filepath + ".faiss")

        # Load stamp database
        with open(filepath + ".json", 'r') as f:
            self.stamp_database = json.load(f)

        print(f"Database loaded from {filepath}")


def main():
    """Demo of the stamp matching system"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    dataset = StampDataset("./images/original")
    subset = dataset.get_random_subset(100)

    # Split for testing
    train_size = int(0.8 * len(subset))
    train_stamps = subset[:train_size]
    test_stamps = subset[train_size:]

    print(
        f"Using {len(train_stamps)} stamps for database, {len(test_stamps)} for testing")

    # Create matcher
    matcher = StampMatcher(
        "./saved_models/stamp_encoder_subset_best.pth", device=device)

    # Create database embeddings
    embeddings = matcher.create_database_embeddings(train_stamps)

    # Save database
    matcher.save_database("./saved_models/stamp_database_subset")

    # Evaluate on test set
    results = matcher.evaluate_on_test_set(
        test_stamps, top_k_values=[1, 5, 10])

    # Test with a specific image
    if test_stamps:
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
