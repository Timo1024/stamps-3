from app.models import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP

class User(Base):
    __tablename__ = "users"

    userid = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP")
