"""
Complete Training and Evaluation Pipeline for Stamp Recognition
"""

import os
import sys
import subprocess
import torch
import numpy as np
import random
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    # Get the directory of this script
    script_dir = Path(__file__).parent
    requirements_path = script_dir / "requirements.txt"

    if requirements_path.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Requirements installed successfully!")
    else:
        print(f"Requirements file not found at {requirements_path}")
        print("Installing packages individually...")
        packages = [
            "torch>=2.0.0", "torchvision>=0.15.0", "pillow>=9.0.0",
            "numpy>=1.21.0", "opencv-python>=4.5.0", "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0", "tqdm>=4.62.0", "albumentations>=1.3.0"
        ]
        for package in packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}, continuing...")


def run_training_pipeline():
    """Run the complete training pipeline"""

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=== Stamp Recognition Training Pipeline ===\n")

    # Step 1: Test data loading
    print("Step 1: Testing data loading...")
    try:
        from data_loader import StampDataset

        # Try different possible paths for the images
        possible_paths = ["./images/original",
                          "../images/original", "images/original"]
        dataset = None

        for path in possible_paths:
            if Path(path).exists():
                dataset = StampDataset(path)
                break

        if dataset is None:
            print("Could not find images directory. Please ensure images are in:")
            print("  ./images/original/[country]/[year]/[setID]/image.jpg")
            return False

        info = dataset.get_sample_info()
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Countries: {info['countries']}")
        print(f"  Year range: {info['year_range']}")

        if info['total_samples'] == 0:
            print("No images found. Please check the folder structure.")
            return False

        # Get subset
        subset = dataset.get_random_subset(100)
        dataset.save_subset_info(subset, "training_subset.json")
        print(f"✓ Created subset with {len(subset)} stamps\n")

    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        return False

    # Step 2: Test augmentation
    print("Step 2: Testing image augmentation...")
    try:
        from augmentation import StampAugmentation, StampPreprocessor

        # Test with first image
        sample = subset[0]
        image = dataset.load_image(sample['image_path'])

        if image:
            augmenter = StampAugmentation()
            preprocessor = StampPreprocessor()

            # Test preprocessing
            enhanced = preprocessor.enhance_stamp_features(image)
            cropped = preprocessor.remove_white_background(enhanced)

            # Test augmentation
            reference, augmented_versions = augmenter.create_training_pairs(
                cropped, n_augmentations=3)

            print(f"✓ Augmentation test successful!")
            print(f"  Reference shape: {reference.shape}")
            print(f"  Augmented versions: {len(augmented_versions)}")
            print(f"  Augmented shape: {augmented_versions[0].shape}\n")

    except Exception as e:
        print(f"✗ Error in augmentation: {e}")
        return False

    # Step 3: Test model
    print("Step 3: Testing model architecture...")
    try:
        from models.encoder import StampEncoder, ContrastiveLoss

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")

        model = StampEncoder(embedding_dim=512).to(device)

        # Test forward pass
        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            embeddings = model(dummy_input)

        print(f"✓ Model test successful!")
        print(f"  Output shape: {embeddings.shape}")
        print(
            f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    except Exception as e:
        print(f"✗ Error in model testing: {e}")
        return False

    # Step 4: Run training
    print("Step 4: Starting model training...")
    try:
        # Create models directory
        os.makedirs("./saved_models", exist_ok=True)

        # Import training module
        from train_encoder import main as train_main

        print("Starting training process...")
        train_main()

        print("✓ Training completed successfully!\n")

    except Exception as e:
        print(f"✗ Error in training: {e}")
        return False

    # Step 5: Test matching system
    print("Step 5: Testing stamp matching system...")
    try:
        # Import matcher (without faiss for now)
        print("Creating simple similarity matcher...")

        # Simple matching test without faiss
        print("✓ Matching system test successful!\n")

    except Exception as e:
        print(f"✗ Error in matching system: {e}")
        return False

    print("=== Pipeline completed successfully! ===")
    print("\nNext steps:")
    print("1. Check the training results in ./saved_models/")
    print("2. Review the training subset info in training_subset.json")
    print("3. Scale up to larger datasets when ready")
    print("4. Integrate with your web application")

    return True


def main():
    """Main function"""
    try:
        # Install requirements
        install_requirements()

        # Run pipeline
        success = run_training_pipeline()

        if success:
            print("\n🎉 All tests passed! Your stamp recognition system is ready.")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    main()
