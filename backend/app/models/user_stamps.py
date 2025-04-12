from app.models import Base
from sqlalchemy import Column, Integer, Text, ForeignKey, TIMESTAMP

class UserStamp(Base):
    __tablename__ = "user_stamps"

    userid = Column(Integer, ForeignKey("users.userid", ondelete="CASCADE"), primary_key=True)
    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    amount_used = Column(Integer, default=0)
    amount_unused = Column(Integer, default=0)
    amount_minted = Column(Integer, default=0)
    amount_letter_fdc = Column(Integer, default=0)
    note = Column(Text)
    created_at = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP", onupdate="CURRENT_TIMESTAMP")
