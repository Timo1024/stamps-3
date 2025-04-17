from sqlalchemy import Column, Integer, String, Text, Date, Float, ForeignKey, TIMESTAMP, BigInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class Set(Base):
    __tablename__ = "sets"
    
    setid = Column(Integer, primary_key=True)
    country = Column(String(255))
    category = Column(String(255))
    year = Column(Integer)
    url = Column(String(2083))
    name = Column(String(2083))
    set_description = Column(Text)
    
    stamps = relationship("Stamp", back_populates="set")

class Stamp(Base):
    __tablename__ = "stamps"
    
    stampid = Column(Integer, primary_key=True)  # Removed autoincrement=True
    number = Column(String(255))
    type = Column(String(255))
    denomination = Column(String(255))
    color = Column(String(255))
    description = Column(Text)
    stamps_issued = Column(String(255))
    mint_condition = Column(String(255))
    unused = Column(String(255))
    used = Column(String(255))
    letter_fdc = Column(String(255))
    date_of_issue = Column(Date)
    perforations = Column(String(255))
    sheet_size = Column(String(255))
    designed = Column(String(255))
    engraved = Column(String(255))
    height_width = Column(String(255))
    image_accuracy = Column(Integer)
    perforation_horizontal = Column(Float)
    perforation_vertical = Column(Float)
    perforation_keyword = Column(String(2083))
    value_from = Column(Float)
    value_to = Column(Float)
    number_issued = Column(BigInteger)
    mint_condition_float = Column(Float)
    unused_float = Column(Float)
    used_float = Column(Float)
    letter_fdc_float = Column(Float)
    sheet_size_amount = Column(Float)
    sheet_size_x = Column(Float)
    sheet_size_y = Column(Float)
    sheet_size_note = Column(String(2083))
    height = Column(Float)
    width = Column(Float)
    
    setid = Column(Integer, ForeignKey("sets.setid", ondelete="CASCADE"))
    set = relationship("Set", back_populates="stamps")
    
    themes = relationship("Theme", secondary="stamp_themes", back_populates="stamps")
    images = relationship("Image", secondary="stamp_images", back_populates="stamps")
    colors = relationship("Color", secondary="stamp_colors", back_populates="stamps")

class User(Base):
    __tablename__ = "users"
    
    userid = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    user_stamps = relationship("UserStamp", back_populates="user")

class UserStamp(Base):
    __tablename__ = "user_stamps"
    
    userid = Column(Integer, ForeignKey("users.userid", ondelete="CASCADE"), primary_key=True)
    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    amount_used = Column(Integer, default=0)
    amount_unused = Column(Integer, default=0)
    amount_minted = Column(Integer, default=0)
    amount_letter_fdc = Column(Integer, default=0)
    note = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    user = relationship("User", back_populates="user_stamps")
    stamp = relationship("Stamp")

class Theme(Base):
    __tablename__ = "themes"
    
    themeid = Column(Integer, primary_key=True)  # Removed autoincrement=True
    name = Column(String(255), nullable=False, unique=True)
    
    stamps = relationship("Stamp", secondary="stamp_themes", back_populates="themes")

class StampTheme(Base):
    __tablename__ = "stamp_themes"
    
    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    themeid = Column(Integer, ForeignKey("themes.themeid", ondelete="CASCADE"), primary_key=True)

class Image(Base):
    __tablename__ = "images"
    
    pathid = Column(Integer, primary_key=True)  # Removed autoincrement=True
    path = Column(String(2083), nullable=False)
    
    stamps = relationship("Stamp", secondary="stamp_images", back_populates="images")

class StampImage(Base):
    __tablename__ = "stamp_images"
    
    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    pathid = Column(Integer, ForeignKey("images.pathid", ondelete="CASCADE"), primary_key=True)

class Color(Base):
    __tablename__ = "colors"
    
    colorid = Column(Integer, primary_key=True)  # Removed autoincrement=True
    name = Column(String(255), nullable=False, unique=True)
    
    stamps = relationship("Stamp", secondary="stamp_colors", back_populates="colors")

class StampColor(Base):
    __tablename__ = "stamp_colors"
    
    stampid = Column(Integer, ForeignKey("stamps.stampid", ondelete="CASCADE"), primary_key=True)
    colorid = Column(Integer, ForeignKey("colors.colorid", ondelete="CASCADE"), primary_key=True)
