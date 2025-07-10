# Stamp Encoder Model Testing

This folder contains scripts to test and evaluate your trained stamp encoder model.

## Files Overview

1. **`test_model.py`** - Full-featured testing script with command-line arguments
2. **`test_model_basic.py`** - Simple interactive testing script (no external dependencies)
3. **`test_model_simple.py`** - Interactive testing with sklearn (requires scikit-learn)
4. **`evaluate_model.py`** - Model evaluation script to check embedding quality

## Prerequisites

Make sure you have trained a model first by running:
```bash
python train.py
```

This will create a model file at `./saved_models/stamp_encoder_triplet.pth`

## Quick Start

### 1. Basic Model Evaluation
```bash
python evaluate_model.py
```
This will:
- Test if the model can distinguish between different stamps
- Check consistency of embeddings for the same stamp
- Provide quality metrics

### 2. Simple Interactive Testing
```bash
python test_model_basic.py
```
This will:
- Show a menu to choose testing options
- Allow you to test with specific images or random samples
- Display similarity results with visualizations

### 3. Advanced Testing (with command-line options)
```bash
python test_model.py --help
```

Example commands:
```bash
# Test embedding quality
python test_model.py --test_quality

# Find similar stamps to a specific image
python test_model.py --query_image ./images/original/Aaland/1984/8938/E-i.jpg

# Find top 10 similar stamps
python test_model.py --query_image ./images/original/Aaland/1984/8938/E-i.jpg --top_k 10

# Pre-compute embeddings for faster search
python test_model.py --precompute
```

## What to Expect

### Good Model Performance
- **Self-similarity**: 0.8-0.95 (same stamp with different augmentations)
- **Cross-similarity**: 0.3-0.7 (different stamps)
- **Discrimination gap**: > 0.2 (difference between self and cross similarity)

### Similarity Search Results
- Higher similarity scores (closer to 1.0) indicate more similar stamps
- The model should find stamps that are visually similar or from the same series

## Troubleshooting

### Model Not Found
```
❌ Model not found at ./saved_models/stamp_encoder_triplet.pth
```
**Solution**: Train the model first with `python train.py`

### CUDA Out of Memory
If you get CUDA memory errors, try:
- Reducing batch size in the testing scripts
- Using CPU instead: modify `device='cpu'` in the scripts

### Poor Similarity Results
If similarity scores are low or inconsistent:
- Train the model for more epochs
- Check if the training loss was decreasing
- Ensure you have enough diverse training data

## Example Usage

### Testing with Your Own Images
```python
# In test_model_basic.py, when prompted:
# 1. Choose option 1 (Test with specific image)
# 2. Enter path: ./images/original/YourCountry/Year/StampID/image.jpg
# 3. Enter number of similar images to find: 5
```

### Understanding Results
```
📊 Top 5 similar stamps:
1. similar_stamp_1.jpg: 0.892
2. similar_stamp_2.jpg: 0.847
3. similar_stamp_3.jpg: 0.823
4. similar_stamp_4.jpg: 0.801
5. similar_stamp_5.jpg: 0.789
```

Higher scores mean more similar stamps. Scores above 0.8 typically indicate very similar stamps.

## Tips

1. **Start with `evaluate_model.py`** to check if your model is working correctly
2. **Use `test_model_basic.py`** for interactive testing with visualizations
3. **Use `test_model.py`** for batch processing or automated testing
4. **Look for patterns** in similarity results - stamps from the same country/year should be more similar

## Customization

You can modify the scripts to:
- Change the number of images processed
- Adjust similarity thresholds
- Add different visualization styles
- Export results to files

## Output Files

- Similarity result visualizations are saved as PNG files
- Model evaluation results are printed to console
- You can modify scripts to save detailed results to CSV files
