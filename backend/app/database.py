import os
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DATABASE_URL from environment or use a default with explicit host
DATABASE_URL = os.getenv("DATABASE_URL")

# Add retry logic for database connection
max_retries = 5
retry_count = 0
while retry_count < max_retries:
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as connection:
            print("Database connection successful!")
        break
    except Exception as e:
        retry_count += 1
        print(f"Database connection attempt {retry_count} failed: {e}")
        if retry_count < max_retries:
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Max retries reached. Could not connect to database.")
            # Don't raise here, let the application start anyway
            # and handle connection errors gracefully

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
