import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# Import our custom modules
from data_loader import StampDataset
from augmentation import StampAugmentation, StampPreprocessor
from models.encoder import StampEncoder


def load_model(model_path, device):
    """Load the trained model"""
    model = StampEncoder(embedding_dim=512, backbone='resnet50')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def create_database_embeddings(model, stamps, augmenter, preprocessor, device):
    """Create embeddings for all stamps in the database"""
    print("Creating database embeddings...")

    database_embeddings = []
    database_info = []

    with torch.no_grad():
        for i, stamp_data in enumerate(stamps):
            if i % 20 == 0:
                print(f"Processing {i+1}/{len(stamps)} stamps...")

            # Load and process image
            image = Image.open(stamp_data['image_path']).convert('RGB')
            image = preprocessor.enhance_stamp_features(image)
            image = preprocessor.remove_white_background(image)

            # Create reference version
            reference = augmenter.process_reference(
                image).unsqueeze(0).to(device)

            # Get embedding
            embedding = model(reference)
            database_embeddings.append(embedding.cpu())
            database_info.append(stamp_data)

    database_embeddings = torch.cat(database_embeddings, dim=0)
    print(f"Database ready with {len(database_embeddings)} stamp embeddings")

    return database_embeddings, database_info


def find_matches(query_image_path, model, database_embeddings, database_info,
                 augmenter, preprocessor, device, top_k=5):
    """Find matching stamps for a query image"""

    # Load and process query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = preprocessor.enhance_stamp_features(query_image)
    query_image = preprocessor.remove_white_background(query_image)

    # Simulate user photo (with augmentations)
    user_photo = augmenter.simulate_user_photo(
        query_image).unsqueeze(0).to(device)

    # Get query embedding
    with torch.no_grad():
        query_embedding = model(user_photo).cpu()

    # Compute similarities
    similarities = F.cosine_similarity(
        query_embedding, database_embeddings, dim=1)

    # Get top matches
    top_similarities, top_indices = torch.topk(similarities, k=top_k)

    matches = []
    for i, (sim, idx) in enumerate(zip(top_similarities, top_indices)):
        match_info = database_info[idx.item()].copy()
        match_info['similarity'] = sim.item()
        match_info['rank'] = i + 1
        matches.append(match_info)

    return matches


def demo_stamp_matching():
    """Demonstrate stamp matching with real examples"""
    print("STAMP MATCHING DEMO")
    print("=" * 50)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "images" / "original",
        script_dir / ".." / "images" / "original",
        Path("./images/original"),
    ]

    dataset = None
    for path in possible_paths:
        if path.exists():
            dataset = StampDataset(str(path))
            break

    if dataset is None:
        print("Error: Could not find images directory!")
        return

    # Get subset for demo
    subset = dataset.get_random_subset(100)
    print(f"Using database of {len(subset)} stamps")

    # Load model
    model_path = "./saved_models/stamp_encoder_subset_best.pth"
    model = load_model(model_path, device)

    # Create augmentation and preprocessing
    augmenter = StampAugmentation(target_size=(224, 224))
    preprocessor = StampPreprocessor()

    # Create database embeddings
    database_embeddings, database_info = create_database_embeddings(
        model, subset, augmenter, preprocessor, device
    )

    # Demo: Test with a few random stamps
    print("\nDEMO RESULTS:")
    print("-" * 50)

    for test_idx in range(3):  # Test 3 random stamps
        # Pick a random stamp from our database
        query_stamp = random.choice(subset)

        print(f"\nTest {test_idx + 1}:")
        print(
            f"Query stamp: {query_stamp['country']}/{query_stamp['year']}/{query_stamp['set_id']}")
        print(f"Path: {query_stamp['image_path']}")

        # Find matches
        matches = find_matches(
            query_stamp['image_path'], model, database_embeddings, database_info,
            augmenter, preprocessor, device, top_k=5
        )

        print("Top 5 matches:")
        for match in matches:
            is_correct = match['unique_id'] == query_stamp['unique_id']
            status = "✅ CORRECT" if is_correct else "❌"
            print(f"  {match['rank']}. {match['country']}/{match['year']}/{match['set_id']} "
                  f"(similarity: {match['similarity']:.3f}) {status}")

        # Check if correct match is in top 1
        top1_correct = matches[0]['unique_id'] == query_stamp['unique_id']
        if top1_correct:
            print("  → SUCCESS: Correct match found at rank 1!")
        else:
            # Find where correct match appears
            correct_rank = None
            for match in matches:
                if match['unique_id'] == query_stamp['unique_id']:
                    correct_rank = match['rank']
                    break
            if correct_rank:
                print(f"  → Correct match found at rank {correct_rank}")
            else:
                print(f"  → Correct match not in top 5")

    print(f"\n{'='*50}")
    print("DEMO COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    demo_stamp_matching()
