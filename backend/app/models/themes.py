from app.models import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP

class Theme(Base):
    __tablename__ = "themes"

    themeid = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)