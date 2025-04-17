import os
import pandas as pd
from sqlalchemy.orm import Session
from . import models
from .database import SessionLocal

def is_table_empty(db: Session, model):
    """Check if a table is empty"""
    return db.query(model).count() == 0

def load_csv_to_table(db: Session, model_class, csv_path):
    """Load data from CSV file into specified table"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return False
    
    try:
        # Use pandas for easy CSV handling
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert records
        for record in records:
            # Remove empty/NaN values
            clean_record = {k: v for k, v in record.items() if pd.notna(v)}
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
        # Map models to CSV files
        model_csv_map = {
            models.Set: "table_data/sets.csv",
            models.Stamp: "table_data/stamps.csv",
            # Add other models as needed
        }
        
        for model, csv_file in model_csv_map.items():
            if is_table_empty(db, model):
                print(f"Table {model.__tablename__} is empty. Loading data...")
                load_csv_to_table(db, model, csv_file)
            else:
                print(f"Table {model.__tablename__} already has data. Skipping...")
    finally:
        db.close()
        
    print("Database initialization complete")
