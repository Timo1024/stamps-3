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
                
        # Add other models as needed (Theme, Color, etc.)
        
    finally:
        db.close()
        
    print("Database initialization complete")
