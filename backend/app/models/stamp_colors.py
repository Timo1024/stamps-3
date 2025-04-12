from app.models import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP

class StampColor(Base):
    __tablename__ = "stamp_colors"

    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    colorid = Column(Integer, ForeignKey("colors.colorid", ondelete="CASCADE"), primary_key=True)