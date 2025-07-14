import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class StampAugmentation:
    """
    Class to simulate how a user-taken photo of a stamp might look
    compared to the perfect reference images
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

        # Define augmentation pipeline that simulates real-world conditions
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations (camera angle, distance)
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                p=0.8
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.5),

            # Lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),

            # Color variations
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.6
            ),

            # Blur and noise (camera quality, motion)
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.4),

            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # JPEG compression artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),

            # Resize to target size
            A.Resize(height=self.target_size[0], width=self.target_size[1]),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

        # Minimal augmentation for reference images (just resize and normalize)
        self.reference_pipeline = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def simulate_user_photo(self, image: Image.Image) -> torch.Tensor:
        """
        Apply augmentation to simulate a user-taken photo
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Apply augmentation
        augmented = self.augmentation_pipeline(image=img_array)
        return augmented['image']

    def process_reference(self, image: Image.Image) -> torch.Tensor:
        """
        Process reference image with minimal augmentation
        """
        img_array = np.array(image)
        processed = self.reference_pipeline(image=img_array)
        return processed['image']

    def create_training_pairs(self, image: Image.Image, n_augmentations=5):
        """
        Create multiple augmented versions of the same stamp for training
        """
        reference = self.process_reference(image)
        augmented_versions = []

        for _ in range(n_augmentations):
            aug_img = self.simulate_user_photo(image)
            augmented_versions.append(aug_img)

        return reference, augmented_versions


class StampPreprocessor:
    """
    Additional preprocessing utilities for stamp images
    """

    @staticmethod
    def remove_white_background(image: Image.Image, threshold=240):
        """
        Attempt to remove white background from stamp images
        """
        img_array = np.array(image)

        # Create mask for non-white pixels
        mask = np.all(img_array < threshold, axis=2)

        # Find bounding box of non-white content
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return image

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop to bounding box with small padding
        padding = 5
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(img_array.shape[0], y1 + padding)
        x1 = min(img_array.shape[1], x1 + padding)

        cropped = image.crop((x0, y0, x1, y1))
        return cropped

    @staticmethod
    def enhance_stamp_features(image: Image.Image):
        """
        Enhance stamp features for better recognition
        """
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)

        # Increase contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)

        return image


if __name__ == "__main__":
    # Test augmentation
    from data_loader import StampDataset

    dataset = StampDataset()
    sample = dataset.samples[0]

    print(f"Testing augmentation with: {sample['unique_id']}")

    image = dataset.load_image(sample['image_path'])
    if image:
        augmenter = StampAugmentation()

        # Test preprocessing
        preprocessor = StampPreprocessor()
        enhanced = preprocessor.enhance_stamp_features(image)
        cropped = preprocessor.remove_white_background(enhanced)

        # Test augmentation
        reference, augmented_versions = augmenter.create_training_pairs(
            cropped, n_augmentations=3)

        print(f"Reference tensor shape: {reference.shape}")
        print(f"Number of augmented versions: {len(augmented_versions)}")
        print(f"Augmented tensor shape: {augmented_versions[0].shape}")

        print("Augmentation test completed successfully!")
