import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict

from data.train_dataset import StampDataset
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Test Augmentations (Simulating User Photos)
# ----------------------------------------


def get_user_photo_augmentations():
    """Aggressive augmentations to simulate real user photos"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.GaussianBlur(blur_limit=7, p=0.5),
        A.GaussNoise(var_limit=(20, 80), p=0.6),
        A.MotionBlur(blur_limit=7, p=0.4),
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.9),
        A.Perspective(scale=(0.02, 0.1), p=0.7),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=3, p=0.5),
        A.RandomRain(blur_value=3, brightness_coefficient=0.7, p=0.4),
        A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.6),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_reference_augmentations():
    """Minimal augmentations for reference images"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ----------------------------------------
# 2. Test Dataset Class
# ----------------------------------------


class StampTestDataset:
    def __init__(self, image_root, model, device):
        self.image_root = image_root
        self.model = model
        self.device = device

        # Load dataset
        self.base_dataset = StampDataset(image_root=image_root, transform=None)

        # Create augmentation pipelines
        self.user_transform = get_user_photo_augmentations()
        self.reference_transform = get_reference_augmentations()

        # Generate reference embeddings
        print("📊 Generating reference embeddings...")
        self.reference_embeddings = self._generate_reference_embeddings()

    def _load_raw_image(self, idx):
        """Load raw image without transforms"""
        image_path = self.base_dataset.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _generate_reference_embeddings(self):
        """Generate embeddings for all reference images"""
        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(self.base_dataset)), desc="Processing references"):
                raw_image = self._load_raw_image(idx)
                ref_image = self.reference_transform(image=raw_image)["image"]
                ref_image = ref_image.unsqueeze(0).to(self.device)

                embedding = self.model(ref_image)
                embedding = F.normalize(embedding, p=2, dim=1)

                # Store with image path as key
                image_path = self.base_dataset.image_paths[idx]
                embeddings[image_path] = embedding.cpu()

        return embeddings

    def test_matching(self, num_test_samples=None, num_augmentations_per_image=5):
        """Test exact matching with augmented user photos"""
        if num_test_samples is None:
            num_test_samples = len(self.base_dataset)

        test_indices = random.sample(range(len(self.base_dataset)),
                                     min(num_test_samples, len(self.base_dataset)))

        results = {
            'correct_matches': 0,
            'total_tests': 0,
            'top_k_correct': defaultdict(int),
            'similarity_scores': [],
            'details': []
        }

        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(test_indices, desc="Testing matching"):
                raw_image = self._load_raw_image(idx)
                ground_truth_path = self.base_dataset.image_paths[idx]

                # Test multiple augmentations of the same image
                for aug_idx in range(num_augmentations_per_image):
                    user_image = self.user_transform(image=raw_image)["image"]
                    user_image = user_image.unsqueeze(0).to(self.device)

                    # Get user image embedding
                    user_embedding = self.model(user_image)
                    user_embedding = F.normalize(user_embedding, p=2, dim=1)

                    # Calculate similarities to all reference images
                    similarities = {}
                    for ref_path, ref_embedding in self.reference_embeddings.items():
                        ref_embedding = ref_embedding.to(self.device)
                        similarity = F.cosine_similarity(
                            user_embedding, ref_embedding, dim=1)
                        similarities[ref_path] = similarity.item()

                    # Sort by similarity
                    sorted_matches = sorted(
                        similarities.items(), key=lambda x: x[1], reverse=True)

                    # Check if correct match is in top-k
                    predicted_path = sorted_matches[0][0]
                    is_correct = predicted_path == ground_truth_path

                    if is_correct:
                        results['correct_matches'] += 1

                    # Check top-k accuracy
                    for k in [1, 3, 5, 10]:
                        if ground_truth_path in [match[0] for match in sorted_matches[:k]]:
                            results['top_k_correct'][k] += 1

                    results['total_tests'] += 1
                    results['similarity_scores'].append(sorted_matches[0][1])

                    # Store details for analysis
                    results['details'].append({
                        'ground_truth': ground_truth_path,
                        'predicted': predicted_path,
                        'correct': is_correct,
                        'similarity': sorted_matches[0][1],
                        'top_5_matches': sorted_matches[:5]
                    })

        return results

# ----------------------------------------
# 3. Evaluation and Visualization
# ----------------------------------------


def print_results(results):
    """Print detailed test results"""
    total = results['total_tests']
    correct = results['correct_matches']

    print(f"\n{'='*50}")
    print(f"🎯 EXACT STAMP MATCHING RESULTS")
    print(f"{'='*50}")
    print(f"Total tests: {total}")
    print(f"Correct matches: {correct}")
    print(f"Top-1 Accuracy: {correct/total*100:.2f}%")

    for k in [1, 3, 5, 10]:
        if k in results['top_k_correct']:
            acc = results['top_k_correct'][k]/total*100
            print(f"Top-{k} Accuracy: {acc:.2f}%")

    # Similarity statistics
    similarities = results['similarity_scores']
    print(f"\nSimilarity Score Statistics:")
    print(f"Mean: {np.mean(similarities):.4f}")
    print(f"Std: {np.std(similarities):.4f}")
    print(f"Min: {np.min(similarities):.4f}")
    print(f"Max: {np.max(similarities):.4f}")

    # Error analysis
    errors = [d for d in results['details'] if not d['correct']]
    if errors:
        print(f"\n❌ Error Analysis ({len(errors)} errors):")
        print(
            f"Average similarity for wrong matches: {np.mean([e['similarity'] for e in errors]):.4f}")

        # Show a few examples
        print("\nSample errors:")
        for i, error in enumerate(errors[:3]):
            print(f"  {i+1}. GT: {os.path.basename(error['ground_truth'])}")
            print(
                f"     Predicted: {os.path.basename(error['predicted'])} (sim: {error['similarity']:.4f})")


def plot_similarity_distribution(results, save_path="similarity_distribution.png"):
    """Plot distribution of similarity scores"""
    similarities = results['similarity_scores']
    correct_sims = [d['similarity']
                    for d in results['details'] if d['correct']]
    wrong_sims = [d['similarity']
                  for d in results['details'] if not d['correct']]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Top-1 Similarity Scores')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if correct_sims:
        plt.hist(correct_sims, bins=30, alpha=0.7,
                 color='green', label='Correct Matches')
    if wrong_sims:
        plt.hist(wrong_sims, bins=30, alpha=0.7,
                 color='red', label='Wrong Matches')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Scores: Correct vs Wrong')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Similarity distribution saved to {save_path}")

# ----------------------------------------
# 4. Main Test Function
# ----------------------------------------


def main():
    # Configuration
    model_path = "./saved_models/stamp_encoder_exact_matching.pth"
    image_root = "./images/original"
    embedding_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Using device: {device}")
    print(f"📁 Loading model from: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_exact_matching.py")
        return

    # Load model
    model = StampEncoder(output_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")

    # Create test dataset
    test_dataset = StampTestDataset(image_root, model, device)
    print(f"📊 Found {len(test_dataset.base_dataset)} stamps for testing")

    # Run tests
    print("\n🧪 Starting comprehensive testing...")
    results = test_dataset.test_matching(
        # Test subset for speed
        num_test_samples=min(100, len(test_dataset.base_dataset)),
        num_augmentations_per_image=3  # Multiple augmentations per image
    )

    # Print results
    print_results(results)

    # Plot results
    plot_similarity_distribution(results)

    # Save detailed results
    results_summary = {
        'total_tests': results['total_tests'],
        'correct_matches': results['correct_matches'],
        'accuracy': results['correct_matches'] / results['total_tests'],
        'top_k_accuracy': {k: v/results['total_tests'] for k, v in results['top_k_correct'].items()}
    }

    print(f"\n📄 Test completed!")
    print(
        f"Model performance: {results_summary['accuracy']*100:.1f}% exact matching accuracy")


if __name__ == "__main__":
    main()
