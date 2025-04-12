from app.models import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP

class StampImage(Base):
    __tablename__ = "stamp_images"

    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    pathid = Column(Integer, ForeignKey("images.pathid", ondelete="CASCADE"), primary_key=True)