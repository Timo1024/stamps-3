#!/usr/bin/env python

"""
Simple utility script to manually populate the database.
Run this directly when you need to repopulate the database without restarting the server.
"""

import os
import sys

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the populate functions
from app.populate_tables import (
    populate_sets,
    populate_stamps,
    populate_themes,
    populate_images,
    populate_colors
)

if __name__ == "__main__":
    print("Starting database population process...")
    
    # Run all populate functions
    populate_sets()
    populate_stamps()
    populate_themes()
    populate_images()
    populate_colors()
    
    print("Database population complete!")
