from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import sys

from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/sets",
    tags=["sets"]
)

@router.get("/", response_model=List[schemas.Set])
def read_sets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        print("Executing set query with skip:", skip, "limit:", limit)
        
        # Execute raw SQL first to verify data exists
        result = db.execute(text("SELECT COUNT(*) FROM sets")).scalar()
        print(f"Total sets in database (raw SQL): {result}")
        
        # Get a few rows for debugging
        sample_sets = db.execute(text("SELECT * FROM sets LIMIT 5")).fetchall()
        if sample_sets:
            print(f"Sample set data found: {len(sample_sets)} rows")
            print(f"First row: {sample_sets[0]}")
        else:
            print("No sample data found with raw SQL")
        
        # Then try the ORM query
        sets = db.query(models.Set).offset(skip).limit(limit).all()
        print(f"Query returned {len(sets)} sets via ORM")
        
        return sets
    except Exception as e:
        print(f"Error in read_sets: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{set_id}", response_model=schemas.Set)
def read_set(set_id: int, db: Session = Depends(get_db)):
    db_set = db.query(models.Set).filter(models.Set.setid == set_id).first()
    if db_set is None:
        raise HTTPException(status_code=404, detail="Set not found")
    return db_set

@router.get("/{set_id}/stamps/", response_model=List[schemas.Stamp])
def read_stamps_by_set(set_id: int, db: Session = Depends(get_db)):
    stamps = db.query(models.Stamp).filter(models.Stamp.setid == set_id).all()
    return stamps
