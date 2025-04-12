from app.models import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP

class Image(Base):
    __tablename__ = "images"

    pathid = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(2083), nullable=False)