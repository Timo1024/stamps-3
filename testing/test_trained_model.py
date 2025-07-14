import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import random
from pathlib import Path
import json

# Import our custom modules
from data_loader import StampDataset
from augmentation import StampAugmentation, StampPreprocessor
from models.encoder import StampEncoder
from train_encoder import StampTrainingDataset


def load_trained_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")

    # Create model
    model = StampEncoder(embedding_dim=512, backbone='resnet50')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully!")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (legacy format)")

    model.to(device)
    model.eval()
    return model


def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix"""
    # Normalize embeddings
    normalized = F.normalize(embeddings, p=2, dim=1)
    # Compute similarity matrix
    similarity_matrix = torch.mm(normalized, normalized.t())
    return similarity_matrix


def test_matching_accuracy(model, test_stamps, augmenter, preprocessor, device, num_test_samples=50):
    """Test how well the model can match augmented stamps to their originals"""
    print(f"\nTesting matching accuracy with {num_test_samples} samples...")

    # Select test samples
    test_samples = random.sample(test_stamps, min(
        num_test_samples, len(test_stamps)))

    reference_embeddings = []
    augmented_embeddings = []
    stamp_labels = []

    with torch.no_grad():
        for idx, stamp_data in enumerate(test_samples):
            # Load and process image
            image = Image.open(stamp_data['image_path']).convert('RGB')
            image = preprocessor.enhance_stamp_features(image)
            image = preprocessor.remove_white_background(image)

            # Create reference and augmented versions
            reference = augmenter.process_reference(
                image).unsqueeze(0).to(device)
            augmented = augmenter.simulate_user_photo(
                image).unsqueeze(0).to(device)

            # Get embeddings
            ref_emb = model(reference)
            aug_emb = model(augmented)

            reference_embeddings.append(ref_emb)
            augmented_embeddings.append(aug_emb)
            stamp_labels.append(idx)

    # Stack embeddings
    ref_embeddings = torch.cat(reference_embeddings, dim=0)
    aug_embeddings = torch.cat(augmented_embeddings, dim=0)

    # Compute similarities
    similarity_matrix = compute_similarity_matrix(
        torch.cat([ref_embeddings, aug_embeddings], dim=0)
    )

    # Extract cross-similarities (augmented vs reference)
    cross_sim = similarity_matrix[num_test_samples:, :num_test_samples]

    # Test accuracy
    correct_matches = 0
    top3_matches = 0
    top5_matches = 0

    for i in range(num_test_samples):
        # Get similarities for this augmented image vs all references
        similarities = cross_sim[i]

        # Get top matches
        _, top_indices = torch.topk(similarities, k=5)

        # Check if correct match is in top positions
        if top_indices[0] == i:
            correct_matches += 1
        if i in top_indices[:3]:
            top3_matches += 1
        if i in top_indices[:5]:
            top5_matches += 1

    # Calculate accuracies
    top1_accuracy = correct_matches / num_test_samples
    top3_accuracy = top3_matches / num_test_samples
    top5_accuracy = top5_matches / num_test_samples

    print(
        f"Top-1 Accuracy: {top1_accuracy:.3f} ({correct_matches}/{num_test_samples})")
    print(
        f"Top-3 Accuracy: {top3_accuracy:.3f} ({top3_matches}/{num_test_samples})")
    print(
        f"Top-5 Accuracy: {top5_accuracy:.3f} ({top5_matches}/{num_test_samples})")

    return top1_accuracy, top3_accuracy, top5_accuracy


def analyze_embeddings(model, test_stamps, augmenter, preprocessor, device, num_samples=20):
    """Analyze the quality of embeddings"""
    print(f"\nAnalyzing embedding quality with {num_samples} samples...")

    test_samples = random.sample(
        test_stamps, min(num_samples, len(test_stamps)))

    same_stamp_similarities = []
    different_stamp_similarities = []

    with torch.no_grad():
        for i, stamp_data in enumerate(test_samples):
            # Load and process image
            image = Image.open(stamp_data['image_path']).convert('RGB')
            image = preprocessor.enhance_stamp_features(image)
            image = preprocessor.remove_white_background(image)

            # Create multiple versions
            reference = augmenter.process_reference(
                image).unsqueeze(0).to(device)
            augmented1 = augmenter.simulate_user_photo(
                image).unsqueeze(0).to(device)
            augmented2 = augmenter.simulate_user_photo(
                image).unsqueeze(0).to(device)

            # Get embeddings
            ref_emb = model(reference)
            aug1_emb = model(augmented1)
            aug2_emb = model(augmented2)

            # Compute similarities within same stamp
            sim_ref_aug1 = F.cosine_similarity(ref_emb, aug1_emb, dim=1).item()
            sim_ref_aug2 = F.cosine_similarity(ref_emb, aug2_emb, dim=1).item()
            sim_aug1_aug2 = F.cosine_similarity(
                aug1_emb, aug2_emb, dim=1).item()

            same_stamp_similarities.extend(
                [sim_ref_aug1, sim_ref_aug2, sim_aug1_aug2])

            # Compute similarities with other stamps
            for j, other_stamp_data in enumerate(test_samples):
                if i != j:
                    other_image = Image.open(
                        other_stamp_data['image_path']).convert('RGB')
                    other_image = preprocessor.enhance_stamp_features(
                        other_image)
                    other_image = preprocessor.remove_white_background(
                        other_image)
                    other_ref = augmenter.process_reference(
                        other_image).unsqueeze(0).to(device)
                    other_emb = model(other_ref)

                    sim_different = F.cosine_similarity(
                        ref_emb, other_emb, dim=1).item()
                    different_stamp_similarities.append(sim_different)

    # Calculate statistics
    same_mean = np.mean(same_stamp_similarities)
    same_std = np.std(same_stamp_similarities)
    diff_mean = np.mean(different_stamp_similarities)
    diff_std = np.std(different_stamp_similarities)

    print(f"Same stamp similarity: {same_mean:.3f} ± {same_std:.3f}")
    print(f"Different stamp similarity: {diff_mean:.3f} ± {diff_std:.3f}")
    print(f"Separation margin: {same_mean - diff_mean:.3f}")

    return same_mean, diff_mean, same_mean - diff_mean


def test_inference_speed(model, device, num_iterations=100):
    """Test inference speed"""
    print(f"\nTesting inference speed with {num_iterations} iterations...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Time inference
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = torch.cuda.Event(
        enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(
        enable_timing=True) if device.type == 'cuda' else None

    if device.type == 'cuda':
        start_time.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(
            end_time) / 1000.0  # Convert to seconds
    else:
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        elapsed_time = time.time() - start_time

    avg_inference_time = elapsed_time / \
        num_iterations * 1000  # Convert to milliseconds
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Throughput: {1000 / avg_inference_time:.1f} images/second")

    return avg_inference_time


def main():
    """Main testing function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading stamp dataset...")
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "images" / "original",
        script_dir / ".." / "images" / "original",
        Path("./images/original"),
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

    # Get subset for testing
    subset = dataset.get_random_subset(100)
    print(f"Using subset of {len(subset)} stamps for testing")

    # Split into train/test (use same split as training)
    random.shuffle(subset)
    split_idx = int(0.8 * len(subset))
    train_stamps = subset[:split_idx]
    test_stamps = subset[split_idx:]

    print(f"Test stamps: {len(test_stamps)}")

    # Create augmentation and preprocessing
    augmenter = StampAugmentation(target_size=(224, 224))
    preprocessor = StampPreprocessor()

    # Load best model
    best_model_path = "./saved_models/stamp_encoder_subset_best.pth"
    model = load_trained_model(best_model_path, device)

    print(f"\n{'='*60}")
    print("STAMP ENCODER MODEL EVALUATION")
    print(f"{'='*60}")

    # Test 1: Matching accuracy
    top1, top3, top5 = test_matching_accuracy(
        model, test_stamps, augmenter, preprocessor, device, num_test_samples=len(
            test_stamps)
    )

    # Test 2: Embedding quality analysis
    same_sim, diff_sim, margin = analyze_embeddings(
        model, test_stamps, augmenter, preprocessor, device, num_samples=len(
            test_stamps)
    )

    # Test 3: Inference speed
    avg_time = test_inference_speed(model, device)

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {len(subset)} stamps ({len(test_stamps)} for testing)")
    print(f"Model: ResNet-50 backbone, 512-dim embeddings")
    print(f"")
    print(f"ACCURACY METRICS:")
    print(f"  Top-1 Accuracy: {top1:.1%}")
    print(f"  Top-3 Accuracy: {top3:.1%}")
    print(f"  Top-5 Accuracy: {top5:.1%}")
    print(f"")
    print(f"EMBEDDING QUALITY:")
    print(f"  Same stamp similarity: {same_sim:.3f}")
    print(f"  Different stamp similarity: {diff_sim:.3f}")
    print(f"  Separation margin: {margin:.3f}")
    print(f"")
    print(f"PERFORMANCE:")
    print(f"  Inference time: {avg_time:.1f} ms")
    print(f"  Throughput: {1000/avg_time:.0f} images/sec")

    # Quality assessment
    print(f"\n{'='*60}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*60}")

    if top1 >= 0.8:
        print("✅ EXCELLENT: Top-1 accuracy >= 80%")
    elif top1 >= 0.6:
        print("✅ GOOD: Top-1 accuracy >= 60%")
    elif top1 >= 0.4:
        print("⚠️  FAIR: Top-1 accuracy >= 40%")
    else:
        print("❌ POOR: Top-1 accuracy < 40%")

    if margin >= 0.2:
        print("✅ EXCELLENT: Strong embedding separation")
    elif margin >= 0.1:
        print("✅ GOOD: Adequate embedding separation")
    else:
        print("⚠️  FAIR: Weak embedding separation")

    if avg_time <= 50:
        print("✅ EXCELLENT: Very fast inference")
    elif avg_time <= 100:
        print("✅ GOOD: Fast inference")
    else:
        print("⚠️  FAIR: Slower inference")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
