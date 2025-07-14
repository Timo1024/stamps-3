#!/usr/bin/env python3
"""
Simple training launcher with correct paths
"""

import os
import sys
from pathlib import Path

# Set working directory to testing folder
testing_dir = Path(__file__).parent
os.chdir(testing_dir)

print(f"Working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

# Start training
print("Starting stamp encoder training...")
print("=" * 50)

try:
    # Import and run training
    exec(open('train_encoder.py').read())
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
