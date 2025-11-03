# app/models.py
from __future__ import annotations
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, UniqueConstraint, Index, JSON, DECIMAL
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)  
    username = Column(String, unique=True)
    email = Column(String, unique=True)                         
    password_hash = Column(String)                              
    role = Column(String)                                    
    created_at = Column(DateTime, default=datetime.utcnow)
    google_identities = relationship("GoogleIdentity", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("SessionRow", back_populates="user", cascade="all, delete-orphan")
    tablatures = relationship("Tablature", back_populates="user", cascade="all, delete-orphan")
    warmups = relationship("WarmupSession", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")

class GoogleIdentity(Base):
    __tablename__ = "google_identities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    google_sub = Column(String, nullable=False, unique=True)  
    email = Column(String)
    email_verified = Column(Boolean, nullable=False, default=False)
    picture_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="google_identities")

class SessionRow(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="sessions")

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    video_path = Column(String)
    status = Column(String)
    submitted_at = Column(DateTime)
    completed_at = Column(DateTime)
    user = relationship("User", back_populates="jobs")
    result = relationship("AnalysisResult", back_populates="job", uselist=False, cascade="all, delete-orphan")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    duration_seconds = Column(DECIMAL, nullable=False)
    overall_score  = Column(DECIMAL)
    posture_score  = Column(DECIMAL)
    rhythmic_score = Column(DECIMAL)
    harmonic_score = Column(DECIMAL)
    posture_json   = Column(JSON) 
    rhythmic_json  = Column(JSON)  
    harmonic_json  = Column(JSON)  
    created_at = Column(DateTime, default=datetime.utcnow)
    job = relationship("Job", back_populates="result")

class Tablature(Base):
    __tablename__ = "tablatures"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title       = Column(String)
    difficulty  = Column(String)
    duration    = Column(String)  
    genre       = Column(String)
    description = Column(Text)
    tabs_json       = Column(JSON, nullable=False) 
    warmup_seq_json = Column(JSON, nullable=False) 
    steps_count     = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="tablatures")
    warmup_sessions = relationship("WarmupSession", back_populates="tablature")
    __table_args__ = (
        Index("ix_tablatures_user_created", "user_id", "created_at"),
    )

class WarmupSession(Base):
    __tablename__ = "warmup_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    tablature_id = Column(Integer, ForeignKey("tablatures.id", ondelete="SET NULL"), index=True)
    duration_seconds  = Column(DECIMAL)
    performance_score = Column(DECIMAL)
    total_steps    = Column(Integer, nullable=False, default=0)
    correct_steps  = Column(Integer, nullable=False, default=0)
    incorrect_steps= Column(Integer, nullable=False, default=0)
    timeouts       = Column(Integer, nullable=False, default=0)
    results_flags  = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="warmups")
    tablature = relationship("Tablature", back_populates="warmup_sessions")
