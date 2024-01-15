from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class TextData(Base):
    __tablename__ = "text_data"
    
    id = Column(Integer, primary_key=True, index=True)
    text_content = Column(Text, nullable=False)
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    labels = relationship("Label", back_populates="text_data")

class Label(Base):
    __tablename__ = "labels"
    
    id = Column(Integer, primary_key=True, index=True)
    text_data_id = Column(Integer, ForeignKey("text_data.id"))
    category = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    text_data = relationship("TextData", back_populates="labels")
    reviews = relationship("Review", back_populates="label")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    label_id = Column(Integer, ForeignKey("labels.id"))
    action = Column(String(50), nullable=False)  # approve, reject, modify
    reviewer_name = Column(String(100), nullable=False)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    label = relationship("Label", back_populates="reviews")

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow) 