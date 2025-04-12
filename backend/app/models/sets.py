from app.models import Base
from sqlalchemy import Column, Integer, String, Text

class Set(Base):
    __tablename__ = "sets"

    setid = Column(Integer, primary_key=True)
    country = Column(String(255))
    category = Column(String(255))
    year = Column(Integer)
    url = Column(String(2083))
    name = Column(String(2083))
    set_description = Column(Text)
