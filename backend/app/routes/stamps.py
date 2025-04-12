from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.stamps import Stamp
# from app.models.pydanic_stamps import StampCreate, StampResponse
from app.__init__ import get_db

router = APIRouter()

# Get all stamps
# @router.get("/stamps/", response_model=list[StampResponse])
# def get_stamps(db: Session = Depends(get_db)):
#     return db.query(Stamp).all()

# Create a new stamp
# @router.post("/stamps/",response_model=StampResponse)
# def create_stamp(stamp: StampCreate, db: Session = Depends(get_db)):
#     new_stamp = Stamp(**stamp.dict())
#     db.add(new_stamp)
#     db.commit()
#     db.refresh(new_stamp)
#     return new_stamp

# Get a stamp by ID
# @router.get("/stamps/{stampid}", response_model=StampResponse)
# def get_stamp(stampid: int, db: Session = Depends(get_db)):
#     stamp = db.query(Stamp).filter(Stamp.stampid == stampid).first()
#     if not stamp:
#         raise HTTPException(status_code=404, detail="Stamp not found")
#     return stamp
