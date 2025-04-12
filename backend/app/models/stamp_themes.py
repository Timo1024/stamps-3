from app.models import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP

class StampTheme(Base):
    __tablename__ = "stamp_themes"

    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    themeid = Column(Integer, ForeignKey("themes.themeid", ondelete="CASCADE"), primary_key=True)