import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from collections import defaultdict

from data.train_dataset import StampDataset
from models.encoder import StampEncoder

# ----------------------------------------
# 1. Test Augmentations (Simulating Human Photos)
# ----------------------------------------


def get_human_photo_augmentations():
    """Very aggressive augmentations to simulate real human-taken photos"""
    return A.Compose([
        # Lighting variations (poor lighting, flash, shadows)
        A.RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=0.95),
        A.HueSaturationValue(hue_shift_limit=25,
                             sat_shift_limit=40, val_shift_limit=30, p=0.9),
        A.RandomGamma(gamma_limit=(50, 150), p=0.7),
        A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.6),

        # Camera quality issues
        A.GaussianBlur(blur_limit=7, p=0.5),
        A.MotionBlur(blur_limit=9, p=0.4),
        A.GaussNoise(var_limit=(0, 0.05), p=0.7),
        A.ImageCompression(quality_lower=40, quality_upper=95, p=0.6),

        # Geometric distortions (hand-held camera)
        A.Rotate(limit=35, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.95),
        A.Perspective(scale=(0.05, 0.15), p=0.8),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3,
                           rotate_limit=20, p=0.9),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),

        # Environmental effects
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=4, p=0.6),
        A.RandomRain(blur_value=3, brightness_coefficient=0.6, p=0.4),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0,
                         angle_upper=1, num_flare_circles_lower=1,
                         num_flare_circles_upper=3, p=0.2),

        # Color distortions
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.15, p=0.8),
        A.ChannelShuffle(p=0.2),
        A.ToGray(p=0.1),  # Sometimes people take b&w photos

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

# ----------------------------------------
# 2. Advanced Test Dataset Class
# ----------------------------------------


class AdvancedStampTestDataset:
    def __init__(self, image_root, model, device):
        self.image_root = image_root
        self.model = model
        self.device = device

        # Load dataset
        self.base_dataset = StampDataset(image_root=image_root, transform=None)

        # Create augmentation pipelines
        self.human_transform = get_human_photo_augmentations()
        self.reference_transform = get_reference_augmentations()

        # Generate reference embeddings
        print("📊 Generating reference embeddings with advanced model...")
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
            for idx in tqdm(range(len(self.base_dataset)), desc="Generating reference embeddings"):
                raw_image = self._load_raw_image(idx)
                ref_image = self.reference_transform(image=raw_image)["image"]
                ref_image = ref_image.unsqueeze(0).to(self.device)

                embedding = self.model(ref_image)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings[idx] = embedding.cpu().squeeze()

        return embeddings

    def test_matching(self, num_test_samples=None, num_augmentations_per_image=5):
        """Test exact matching with heavily augmented human photos"""
        if num_test_samples is None:
            num_test_samples = len(self.base_dataset)

        test_indices = random.sample(range(len(self.base_dataset)),
                                     min(num_test_samples, len(self.base_dataset)))

        results = {
            'correct_matches': 0,
            'total_tests': 0,
            'top_k_correct': defaultdict(int),
            'similarity_scores': [],
            'confidence_scores': [],
            'details': []
        }

        self.model.eval()
        with torch.no_grad():
            for test_idx in tqdm(test_indices, desc="Testing stamp matching"):
                raw_image = self._load_raw_image(test_idx)
                true_label = test_idx

                # Test multiple augmentations per image
                for aug_num in range(num_augmentations_per_image):
                    # Apply human photo augmentation
                    human_photo = self.human_transform(
                        image=raw_image)["image"]
                    human_photo = human_photo.unsqueeze(0).to(self.device)

                    # Get embedding for human photo
                    query_embedding = self.model(human_photo)
                    query_embedding = F.normalize(
                        query_embedding, p=2, dim=1).cpu().squeeze()

                    # Calculate similarities with all reference images
                    similarities = []
                    for ref_idx in range(len(self.base_dataset)):
                        ref_embedding = self.reference_embeddings[ref_idx]
                        similarity = F.cosine_similarity(
                            query_embedding.unsqueeze(0),
                            ref_embedding.unsqueeze(0)
                        ).item()
                        similarities.append((similarity, ref_idx))

                    # Sort by similarity (highest first)
                    similarities.sort(reverse=True, key=lambda x: x[0])

                    # Check if correct match is in top-k
                    predicted_idx = similarities[0][1]
                    top_similarity = similarities[0][0]

                    is_correct = (predicted_idx == true_label)

                    # Update results
                    results['total_tests'] += 1
                    if is_correct:
                        results['correct_matches'] += 1

                    # Top-k accuracy
                    for k in [1, 3, 5, 10, 20]:
                        if any(sim[1] == true_label for sim in similarities[:k]):
                            results['top_k_correct'][k] += 1

                    results['similarity_scores'].append(top_similarity)

                    # Calculate confidence (gap between top 2 matches)
                    if len(similarities) > 1:
                        confidence = similarities[0][0] - similarities[1][0]
                        results['confidence_scores'].append(confidence)

                    # Store detailed results
                    results['details'].append({
                        'true_idx': true_label,
                        'predicted_idx': predicted_idx,
                        'similarity': top_similarity,
                        'correct': is_correct,
                        'augmentation': aug_num,
                        'top_5_matches': similarities[:5]
                    })

        return results

# ----------------------------------------
# 3. Evaluation and Visualization
# ----------------------------------------


def print_advanced_results(results):
    """Print detailed test results for advanced model"""
    total = results['total_tests']
    correct = results['correct_matches']

    print(f"\n{'='*60}")
    print(f"🎯 ADVANCED STAMP MATCHING RESULTS")
    print(f"{'='*60}")
    print(f"Total tests: {total}")
    print(f"Correct matches: {correct}")
    print(f"🏆 Top-1 Accuracy: {correct/total*100:.2f}%")

    for k in [1, 3, 5, 10, 20]:
        if k in results['top_k_correct']:
            acc = results['top_k_correct'][k]/total*100
            print(f"📊 Top-{k} Accuracy: {acc:.2f}%")

    # Similarity statistics
    similarities = results['similarity_scores']
    confidences = results['confidence_scores']

    print(f"\n📈 Similarity Score Statistics:")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std: {np.std(similarities):.4f}")
    print(f"  Min: {np.min(similarities):.4f}")
    print(f"  Max: {np.max(similarities):.4f}")

    print(f"\n🎯 Confidence Statistics (gap between top 2):")
    print(f"  Mean: {np.mean(confidences):.4f}")
    print(f"  Std: {np.std(confidences):.4f}")
    print(f"  Min: {np.min(confidences):.4f}")
    print(f"  Max: {np.max(confidences):.4f}")

    # Error analysis
    errors = [d for d in results['details'] if not d['correct']]
    corrects = [d for d in results['details'] if d['correct']]

    if errors:
        print(f"\n❌ Error Analysis ({len(errors)} errors out of {total}):")
        print(
            f"  Average similarity for wrong matches: {np.mean([e['similarity'] for e in errors]):.4f}")
        print(
            f"  Average confidence for wrong matches: {np.mean([e.get('top_5_matches', [('', 0), ('', 0)])[0][0] - e.get('top_5_matches', [('', 0), ('', 0)])[1][0] for e in errors if len(e.get('top_5_matches', [])) > 1]):.4f}")

    if corrects:
        print(f"\n✅ Correct Match Analysis ({len(corrects)} correct):")
        print(
            f"  Average similarity for correct matches: {np.mean([c['similarity'] for c in corrects]):.4f}")

    # Show sample errors and successes
    if errors:
        print(f"\n🔍 Sample Errors:")
        for i, error in enumerate(errors[:3]):
            print(
                f"  Error {i+1}: True={error['true_idx']}, Predicted={error['predicted_idx']}, Similarity={error['similarity']:.4f}")

    if corrects:
        print(f"\n🎉 Sample Successes:")
        for i, success in enumerate(corrects[:3]):
            print(
                f"  Success {i+1}: True={success['true_idx']}, Predicted={success['predicted_idx']}, Similarity={success['similarity']:.4f}")


def plot_advanced_results(results, save_path="advanced_results.png"):
    """Plot comprehensive results visualization"""
    plt.figure(figsize=(15, 10))

    # 1. Similarity distribution
    plt.subplot(2, 3, 1)
    similarities = results['similarity_scores']
    correct_sims = [d['similarity']
                    for d in results['details'] if d['correct']]
    wrong_sims = [d['similarity']
                  for d in results['details'] if not d['correct']]

    plt.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Top-1 Similarity Scores')
    plt.grid(True, alpha=0.3)

    # 2. Correct vs Wrong similarities
    plt.subplot(2, 3, 2)
    if correct_sims:
        plt.hist(correct_sims, bins=30, alpha=0.7,
                 color='green', label='Correct Matches')
    if wrong_sims:
        plt.hist(wrong_sims, bins=30, alpha=0.7,
                 color='red', label='Wrong Matches')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity: Correct vs Wrong')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Top-k accuracy
    plt.subplot(2, 3, 3)
    k_values = [1, 3, 5, 10, 20]
    accuracies = [results['top_k_correct'][k]/results['total_tests']
                  * 100 for k in k_values if k in results['top_k_correct']]
    k_values = [k for k in k_values if k in results['top_k_correct']]

    plt.bar(range(len(k_values)), accuracies,
            color='skyblue', edgecolor='navy')
    plt.xlabel('Top-K')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-K Accuracy')
    plt.xticks(range(len(k_values)), [f'Top-{k}' for k in k_values])
    plt.grid(True, alpha=0.3)

    # 4. Confidence scores
    plt.subplot(2, 3, 4)
    confidences = results['confidence_scores']
    plt.hist(confidences, bins=50, alpha=0.7,
             color='orange', edgecolor='black')
    plt.xlabel('Confidence (Top1-Top2 Similarity)')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.grid(True, alpha=0.3)

    # 5. Similarity vs Accuracy
    plt.subplot(2, 3, 5)
    sim_bins = np.linspace(min(similarities), max(similarities), 10)
    bin_accuracies = []
    bin_centers = []

    for i in range(len(sim_bins)-1):
        bin_details = [d for d in results['details']
                       if sim_bins[i] <= d['similarity'] < sim_bins[i+1]]
        if bin_details:
            accuracy = sum(d['correct']
                           for d in bin_details) / len(bin_details) * 100
            bin_accuracies.append(accuracy)
            bin_centers.append((sim_bins[i] + sim_bins[i+1]) / 2)

    plt.plot(bin_centers, bin_accuracies, 'o-', color='purple')
    plt.xlabel('Similarity Score')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Similarity Score')
    plt.grid(True, alpha=0.3)

    # 6. Model performance summary
    plt.subplot(2, 3, 6)
    metrics = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
    values = [results['top_k_correct'][k]/results['total_tests']*100
              for k in [1, 3, 5, 10] if k in results['top_k_correct']]
    metrics = metrics[:len(values)]

    colors = ['red', 'orange', 'yellow', 'green'][:len(values)]
    plt.pie(values, labels=metrics, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.title('Performance Breakdown')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Advanced results visualization saved to {save_path}")

# ----------------------------------------
# 4. Main Test Function
# ----------------------------------------


def main():
    # Configuration
    model_path = "./saved_models/stamp_encoder_advanced.pth"
    image_root = "./images/original"
    embedding_dim = 512  # Advanced model uses 512 dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Using device: {device}")
    print(f"📁 Loading ADVANCED model from: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Advanced model not found at {model_path}")
        print("Please train the advanced model first using train_advanced.py")
        return

    # Load model
    model = StampEncoder(output_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Advanced model loaded successfully")

    # Create test dataset
    test_dataset = AdvancedStampTestDataset(image_root, model, device)
    print(f"📊 Found {len(test_dataset.base_dataset)} stamps for testing")

    # Run comprehensive tests
    print("\n🧪 Starting COMPREHENSIVE testing with human photo simulation...")
    print("⚠️  This will take longer due to intensive augmentations...")

    results = test_dataset.test_matching(
        # Test more samples
        num_test_samples=min(150, len(test_dataset.base_dataset)),
        num_augmentations_per_image=3  # Multiple severe augmentations per image
    )

    # Print results
    print_advanced_results(results)

    # Plot results
    plot_advanced_results(results)

    # Save detailed results
    results_summary = {
        'model_type': 'advanced',
        'embedding_dim': embedding_dim,
        'total_tests': results['total_tests'],
        'correct_matches': results['correct_matches'],
        'top_1_accuracy': results['correct_matches'] / results['total_tests'],
        'top_k_accuracy': {k: v/results['total_tests'] for k, v in results['top_k_correct'].items()},
        'mean_similarity': np.mean(results['similarity_scores']),
        'mean_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
    }

    print(f"\n📄 Advanced model testing completed!")
    print(
        f"🎯 Final Performance: {results_summary['top_1_accuracy']*100:.1f}% exact matching accuracy")
    print(
        f"📈 Top-5 Performance: {results_summary['top_k_accuracy'].get(5, 0)*100:.1f}% top-5 accuracy")

    if results_summary['top_1_accuracy'] > 0.7:
        print("🎉 EXCELLENT! Model shows strong exact matching capability")
    elif results_summary['top_1_accuracy'] > 0.5:
        print("👍 GOOD! Model shows decent exact matching capability")
    elif results_summary['top_1_accuracy'] > 0.3:
        print("⚠️  FAIR! Model needs improvement for reliable matching")
    else:
        print("❌ POOR! Model requires significant improvements")


if __name__ == "__main__":
    main()
