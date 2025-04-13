from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from .. import models
from ..database import get_db, engine

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Welcome to the Stamps API"}

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.get("/diagnostic")
def check_database_counts(db: Session = Depends(get_db)):
    """
    Diagnostic endpoint to check database record counts
    """
    return {
        "sets_count": db.query(models.Set).count(),
        "stamps_count": db.query(models.Stamp).count(),
        "themes_count": db.query(models.Theme).count(),
        "images_count": db.query(models.Image).count(),
        "colors_count": db.query(models.Color).count(),
        "users_count": db.query(models.User).count(),
    }

@router.get("/diagnostic/db")
def database_diagnostic():
    """Check database connectivity and table contents"""
    results = {}
    conn = engine.connect()
    
    try:
        # Get table counts
        tables = ["sets", "stamps", "themes", "images", "colors"]
        for table in tables:
            query = text(f"SELECT COUNT(*) FROM {table}")
            count = conn.execute(query).scalar()
            results[f"{table}_count"] = count
        
        # Get sample data 
        sample_query = text("SELECT setid, country, name FROM sets LIMIT 3")
        samples = [dict(row) for row in conn.execute(sample_query).fetchall()]
        results["sample_sets"] = samples
        
        # Get database version
        version = conn.execute(text("SELECT VERSION()")).scalar()
        results["db_version"] = version
        
        # Get connection info
        results["connection_info"] = str(engine.url)
        
        return results
    finally:
        conn.close()
