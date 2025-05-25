from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/themes",
    tags=["themes"]
)


@router.get("/", response_model=List[schemas.Theme])
def read_themes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    themes = db.query(models.Theme).offset(skip).limit(limit).all()
    return themes


@router.get("/unique", response_model=List[str])
def get_unique(db: Session = Depends(get_db)):
    # routes when loading page to get data for filters
    # get unique countries and categories
    unique_themes = db.query(models.Theme.name).distinct().all()
    return [t[0] for t in unique_themes]
