import os
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np


class StampDataset:
    """
    Dataset class for loading stamp images from the organized folder structure
    """

    def __init__(self, base_path: str = "./images/original"):
        self.base_path = Path(base_path)
        self.samples = []
        self.country_to_idx = {}
        self.idx_to_country = {}
        self.load_dataset()

    def load_dataset(self):
        """Load all stamp images and create mapping"""
        print("Loading dataset...")

        countries = [d for d in self.base_path.iterdir() if d.is_dir()]

        for country_idx, country_dir in enumerate(countries):
            country_name = country_dir.name
            self.country_to_idx[country_name] = country_idx
            self.idx_to_country[country_idx] = country_name

            # Traverse year directories
            for year_dir in country_dir.iterdir():
                if not year_dir.is_dir():
                    continue

                # Traverse set directories
                for set_dir in year_dir.iterdir():
                    if not set_dir.is_dir():
                        continue

                    # Look for image files
                    for img_file in set_dir.iterdir():
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append({
                                'image_path': str(img_file),
                                'country': country_name,
                                'year': year_dir.name,
                                'set_id': set_dir.name,
                                'country_idx': country_idx,
                                'unique_id': f"{country_name}_{year_dir.name}_{set_dir.name}_{img_file.stem}"
                            })

        print(
            f"Loaded {len(self.samples)} stamp images from {len(countries)} countries")

    def get_random_subset(self, n_samples: int = 100) -> List[Dict]:
        """Get a random subset of stamps for faster training"""
        if n_samples >= len(self.samples):
            return self.samples
        return random.sample(self.samples, n_samples)

    def load_image(self, image_path: str) -> Image.Image:
        """Load and return PIL Image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_sample_info(self) -> Dict:
        """Get information about the dataset"""
        countries = set(sample['country'] for sample in self.samples)
        years = set(sample['year'] for sample in self.samples)

        return {
            'total_samples': len(self.samples),
            'countries': len(countries),
            'years': len(years),
            'country_list': sorted(list(countries)),
            'year_range': f"{min(years)} - {max(years)}"
        }

    def save_subset_info(self, subset: List[Dict], filename: str = "subset_info.json"):
        """Save subset information for reproducibility"""
        subset_info = {
            'samples': subset,
            'total_count': len(subset),
            'countries': list(set(s['country'] for s in subset)),
            'years': list(set(s['year'] for s in subset))
        }

        with open(filename, 'w') as f:
            json.dump(subset_info, f, indent=2)

        print(f"Subset info saved to {filename}")


if __name__ == "__main__":
    # Test the dataset loader
    dataset = StampDataset()
    info = dataset.get_sample_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get a random subset
    subset = dataset.get_random_subset(100)
    dataset.save_subset_info(subset)

    print(f"\nRandom subset of {len(subset)} samples created")
    print("Countries in subset:", set(s['country'] for s in subset))
