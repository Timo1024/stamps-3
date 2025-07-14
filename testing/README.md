# Stamp Recognition System

This folder contains a complete machine learning pipeline for stamp recognition and matching.

## Overview

The system is designed to:
1. Load stamp images from the organized folder structure
2. Create augmented versions that simulate user-taken photos
3. Train a deep learning encoder to extract features
4. Match user-uploaded images to stamps in the database

## Files

### Core Components
- `data_loader.py` - Dataset loading and management
- `augmentation.py` - Image preprocessing and augmentation
- `models/encoder.py` - Neural network architecture
- `train_encoder.py` - Training pipeline
- `simple_matcher.py` - Stamp matching without heavy dependencies
- `stamp_matcher.py` - Advanced matching with FAISS (requires faiss-cpu)

### Utilities
- `run_pipeline.py` - Complete automated pipeline
- `requirements.txt` - Python dependencies

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python run_pipeline.py
   ```

3. **Or run steps individually:**
   ```bash
   # Train the model
   python train_encoder.py
   
   # Test the matcher
   python simple_matcher.py
   ```

## How It Works

### 1. Data Loading
The system automatically discovers all stamp images in the folder structure:
```
images/original/[country]/[year]/[setID]/image.jpg
```

### 2. Augmentation
Each stamp image is augmented to simulate real-world conditions:
- Geometric transformations (rotation, scaling, perspective)
- Lighting variations (brightness, contrast, gamma)
- Color changes (hue, saturation)
- Blur and noise (camera quality simulation)
- JPEG compression artifacts

### 3. Model Architecture
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Encoder**: Feature extraction to 512-dimensional embeddings
- **Loss**: Contrastive + Triplet loss for similarity learning
- **Normalization**: L2 normalized embeddings for cosine similarity

### 4. Training
- Uses subset of 100 images for fast development
- 80/20 train/validation split
- Multiple augmented versions per stamp
- Early stopping and learning rate scheduling

### 5. Matching
- Extract features from user image
- Compare with database using cosine similarity
- Return top-k most similar stamps

## Performance Expectations

With the current subset training (100 stamps):
- **Top-1 Accuracy**: ~60-80% (exact match)
- **Top-5 Accuracy**: ~85-95% (correct stamp in top 5)
- **Top-10 Accuracy**: ~95-99% (correct stamp in top 10)

For production with 1M stamps, you'll need:
- Larger training set
- More sophisticated architecture
- FAISS index for fast similarity search
- GPU training and inference

## Scaling Up

To scale to 1 million stamps:

1. **Increase training data**:
   - Remove the 100-sample limit in `data_loader.py`
   - Use data loading with multiple workers
   - Consider distributed training

2. **Optimize architecture**:
   - Try EfficientNet or Vision Transformer backbones
   - Increase embedding dimensions
   - Add attention mechanisms

3. **Use FAISS for fast search**:
   - Install `faiss-cpu` or `faiss-gpu`
   - Use `stamp_matcher.py` instead of `simple_matcher.py`
   - Consider approximate search methods

4. **Hardware requirements**:
   - GPU with 8GB+ VRAM for training
   - 16GB+ RAM for large datasets
   - SSD storage for fast data loading

## File Structure

```
testing/
├── data_loader.py          # Dataset management
├── augmentation.py         # Image preprocessing
├── train_encoder.py        # Training pipeline
├── simple_matcher.py       # Basic matching
├── stamp_matcher.py        # Advanced matching (FAISS)
├── run_pipeline.py         # Automated pipeline
├── requirements.txt        # Dependencies
├── models/
│   └── encoder.py         # Neural network models
├── saved_models/          # Trained models (created during training)
└── images/
    └── original/          # Your stamp images
        └── [country]/
            └── [year]/
                └── [setID]/
                    └── image.jpg
```

## Next Steps

1. **Run the pipeline** to train your first model
2. **Test with different images** to see performance
3. **Adjust augmentation** parameters if needed
4. **Scale up** when ready for production
5. **Integrate** with your web application

The system is designed to be modular and easy to extend. You can modify individual components without affecting the rest of the pipeline.
