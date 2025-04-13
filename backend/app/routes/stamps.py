from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/stamps",
    tags=["stamps"]
)

@router.get("/", response_model=List[schemas.Stamp])
def read_stamps(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    stamps = db.query(models.Stamp).offset(skip).limit(limit).all()
    return stamps

@router.get("/{stamp_id}", response_model=schemas.Stamp)
def read_stamp(stamp_id: int, db: Session = Depends(get_db)):
    stamp = db.query(models.Stamp).filter(models.Stamp.stampid == stamp_id).first()
    if stamp is None:
        raise HTTPException(status_code=404, detail="Stamp not found")
    return stamp
