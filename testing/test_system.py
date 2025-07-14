"""
Quick test script for the stamp recognition system
"""

import torch
import numpy as np
import random
from pathlib import Path


def test_basic_functionality():
    """Test basic functionality without installing requirements"""

    print("=== Quick Stamp Recognition Test ===\n")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Test 1: Check PyTorch
    print("Test 1: PyTorch Installation")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ PyTorch {torch.__version__} loaded successfully")
        print(f"✓ Using device: {device}")

        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
        print()
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

    # Test 2: Data Loading
    print("Test 2: Data Loading")
    try:
        from data_loader import StampDataset

        # Try to find images
        possible_paths = ["./images/original",
                          "../images/original", "images/original"]
        dataset = None

        for path in possible_paths:
            if Path(path).exists():
                print(f"Found images at: {path}")
                dataset = StampDataset(path)
                break

        if dataset is None:
            print("✗ No images found. Please ensure you have images in:")
            print("  ./images/original/[country]/[year]/[setID]/image.jpg")
            return False

        info = dataset.get_sample_info()
        print(f"✓ Loaded {info['total_samples']} stamp images")
        print(f"✓ Found {info['countries']} countries")

        if info['total_samples'] < 10:
            print(
                "⚠️  Warning: Very few images found. Consider adding more for better training.")

        # Get a small subset for testing
        subset = dataset.get_random_subset(10)
        print(f"✓ Created test subset with {len(subset)} stamps")
        print()

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

    # Test 3: Image Processing
    print("Test 3: Image Processing")
    try:
        from augmentation import StampAugmentation, StampPreprocessor
        from PIL import Image

        # Test with first image
        if subset:
            sample = subset[0]
            image = dataset.load_image(sample['image_path'])

            if image:
                print(f"✓ Loaded test image: {sample['unique_id']}")
                print(f"✓ Original size: {image.size}")

                # Test preprocessing
                preprocessor = StampPreprocessor()
                enhanced = preprocessor.enhance_stamp_features(image)
                cropped = preprocessor.remove_white_background(enhanced)
                print(f"✓ Preprocessing successful")

                # Test augmentation
                augmenter = StampAugmentation(target_size=(224, 224))
                reference = augmenter.process_reference(cropped)
                augmented = augmenter.simulate_user_photo(cropped)

                print(f"✓ Augmentation successful")
                print(f"✓ Reference tensor shape: {reference.shape}")
                print(f"✓ Augmented tensor shape: {augmented.shape}")
                print()
            else:
                print("✗ Could not load test image")
                return False

    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False

    # Test 4: Model Architecture
    print("Test 4: Model Architecture")
    try:
        from models.encoder import StampEncoder, ContrastiveLoss

        model = StampEncoder(embedding_dim=512).to(device)
        print(f"✓ Model created successfully")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            embeddings = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {embeddings.shape}")
        print(
            f"✓ Embedding norm: {torch.norm(embeddings, dim=1).mean().item():.4f}")

        # Test loss
        loss_fn = ContrastiveLoss()
        labels = torch.tensor([0, 1]).to(device)
        loss = loss_fn(embeddings, labels)
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        print()

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

    print("🎉 All basic tests passed!")
    print("\nYour stamp recognition system is ready for training!")
    print("\nNext steps:")
    print("1. Run: python train_encoder.py (for full training)")
    print("2. Or modify train_encoder.py to use smaller batch sizes if memory is limited")
    print("3. Check the saved models in ./saved_models/ after training")

    return True


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if not success:
            print("\n❌ Some tests failed. Please check the error messages above.")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
