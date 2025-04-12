from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os

from . import models, schemas
from .database import engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Stamps API", description="API for stamp collectors")

@app.get("/")
def read_root():
    return {"message": "Welcome to Stamps API"}

# Sets endpoints
@app.get("/sets/", response_model=List[schemas.Set])
def read_sets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sets = db.query(models.Set).offset(skip).limit(limit).all()
    return sets

@app.get("/sets/{set_id}", response_model=schemas.Set)
def read_set(set_id: int, db: Session = Depends(get_db)):
    db_set = db.query(models.Set).filter(models.Set.setid == set_id).first()
    if db_set is None:
        raise HTTPException(status_code=404, detail="Set not found")
    return db_set

# Stamps endpoints
@app.get("/stamps/", response_model=List[schemas.Stamp])
def read_stamps(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    stamps = db.query(models.Stamp).offset(skip).limit(limit).all()
    return stamps

@app.get("/stamps/{stamp_id}", response_model=schemas.Stamp)
def read_stamp(stamp_id: int, db: Session = Depends(get_db)):
    stamp = db.query(models.Stamp).filter(models.Stamp.stampid == stamp_id).first()
    if stamp is None:
        raise HTTPException(status_code=404, detail="Stamp not found")
    return stamp

@app.get("/sets/{set_id}/stamps/", response_model=List[schemas.Stamp])
def read_stamps_by_set(set_id: int, db: Session = Depends(get_db)):
    stamps = db.query(models.Stamp).filter(models.Stamp.setid == set_id).all()
    return stamps
