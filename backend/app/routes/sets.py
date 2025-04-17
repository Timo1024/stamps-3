from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/sets",
    tags=["sets"]
)

@router.get("/", response_model=List[schemas.Set])
def read_sets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sets = db.query(models.Set).offset(skip).limit(limit).all()
    return sets

@router.get("/{set_id}", response_model=schemas.Set)
def read_set(set_id: int, db: Session = Depends(get_db)):
    db_set = db.query(models.Set).filter(models.Set.setid == set_id).first()
    if db_set is None:
        raise HTTPException(status_code=404, detail="Set not found")
    return db_set

@router.get("/{set_id}/stamps/", response_model=List[schemas.Stamp])
def read_stamps_by_set(set_id: int, db: Session = Depends(get_db)):
    # Check if set exists
    db_set = db.query(models.Set).filter(models.Set.setid == set_id).first()
    if db_set is None:
        raise HTTPException(status_code=404, detail="Set not found")
    
    # Get stamps for this set
    stamps = db.query(models.Stamp).filter(models.Stamp.setid == set_id).all()
    return stamps
