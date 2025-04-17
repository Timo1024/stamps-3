import os
import pandas as pd
from sqlalchemy.orm import Session
from . import models
from .database import SessionLocal

def is_table_empty(db: Session, model):
    """Check if a table is empty"""
    return db.query(model).count() == 0

def load_csv_to_table(db: Session, model_class, csv_path, column_mapping=None):
    """Load data from CSV file into specified table"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return False
    
    try:
        # Use pandas for easy CSV handling
        df = pd.read_csv(csv_path)
        
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Get model columns for filtering
        valid_columns = model_class.__table__.columns.keys()
        
        # Insert records
        for record in records:
            # Remove empty/NaN values and invalid columns
            clean_record = {k: v for k, v in record.items() if pd.notna(v) and k in valid_columns}
            db_record = model_class(**clean_record)
            db.add(db_record)
        
        db.commit()
        print(f"Successfully loaded data from {csv_path}")
        return True
    except Exception as e:
        db.rollback()
        print(f"Error loading data from {csv_path}: {str(e)}")
        return False

def initialize_data():
    """Initialize database with data from CSV files if tables are empty"""
    print("Checking if database needs initialization...")
    db = SessionLocal()
    try:
        # First load sets (parent table)
        if is_table_empty(db, models.Set):
            print(f"Table sets is empty. Loading data...")
            # Map 'description' in CSV to 'set_description' in model
            column_mapping = {'description': 'set_description'}
            sets_loaded = load_csv_to_table(db, models.Set, "table_data/sets.csv", column_mapping)
            
            # Only load stamps if sets were successfully loaded
            if sets_loaded and is_table_empty(db, models.Stamp):
                print(f"Table stamps is empty. Loading data...")
                load_csv_to_table(db, models.Stamp, "table_data/stamps.csv")
            elif not sets_loaded:
                print("Skipping stamps import because sets import failed")
        else:
            print(f"Table sets already has data. Skipping...")
            
        # Check stamps separately
        if is_table_empty(db, models.Stamp):
            print(f"Table stamps is empty. Loading data...")
            load_csv_to_table(db, models.Stamp, "table_data/stamps.csv")
        else:
            print(f"Table stamps already has data. Skipping...")
        
        # Load themes data if table is empty
        if is_table_empty(db, models.Theme):
            print(f"Table themes is empty. Loading data...")
            load_csv_to_table(db, models.Theme, "table_data/themes.csv")
        else:
            print(f"Table themes already has data. Skipping...")
        
        # Load images data if table is empty
        if is_table_empty(db, models.Image):
            print(f"Table images is empty. Loading data...")
            load_csv_to_table(db, models.Image, "table_data/images.csv")
        else:
            print(f"Table images already has data. Skipping...")
            
        # Load colors data if table is empty
        if is_table_empty(db, models.Color):
            print(f"Table colors is empty. Loading data...")
            load_csv_to_table(db, models.Color, "table_data/colors.csv")
        else:
            print(f"Table colors already has data. Skipping...")
            
        # Load relationship tables after parent tables
        
        # Load stamp_themes relationships if table is empty
        if is_table_empty(db, models.StampTheme):
            print(f"Table stamp_themes is empty. Loading data...")
            load_csv_to_table(db, models.StampTheme, "table_data/stamp_themes.csv")
        else:
            print(f"Table stamp_themes already has data. Skipping...")
            
        # Load stamp_images relationships if table is empty
        if is_table_empty(db, models.StampImage):
            print(f"Table stamp_images is empty. Loading data...")
            load_csv_to_table(db, models.StampImage, "table_data/stamp_images.csv")
        else:
            print(f"Table stamp_images already has data. Skipping...")
            
        # Load stamp_colors relationships if table is empty
        if is_table_empty(db, models.StampColor):
            print(f"Table stamp_colors is empty. Loading data...")
            load_csv_to_table(db, models.StampColor, "table_data/stamp_colors.csv")
        else:
            print(f"Table stamp_colors already has data. Skipping...")
            
    finally:
        db.close()
        
    print("Database initialization complete")
