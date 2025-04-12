from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel

# Set schemas
class SetBase(BaseModel):
    country: Optional[str] = None
    category: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    name: Optional[str] = None
    set_description: Optional[str] = None

class SetCreate(SetBase):
    pass

class Set(SetBase):
    setid: int
    
    class Config:
        from_attributes = True

# Stamp schemas
class StampBase(BaseModel):
    number: Optional[str] = None
    type: Optional[str] = None
    denomination: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    date_of_issue: Optional[date] = None
    setid: int

class StampCreate(StampBase):
    pass

class Stamp(StampBase):
    stampid: int
    
    class Config:
        from_attributes = True

# User schemas
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    userid: int
    created_at: datetime
    
    class Config:
        from_attributes = True
