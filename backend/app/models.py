#models.py
from backend.app.database import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, Float
from sqlalchemy.sql import func
import json

class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    surname = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LaptopSetup(Base):
    __tablename__ = 'laptop_setup'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    brand = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LaptopSpecs(Base):
    __tablename__ = 'laptop_specs'

    id = Column(Integer, primary_key=True, index=True)
    laptop_setup_id = Column(Integer, ForeignKey("laptop_setup.id"), unique=True, nullable=False)
    cpu = Column(String(255), nullable=True)
    gpu = Column(String(255), nullable=True)
    ram = Column(String(100), nullable=True)
    storage = Column(String(255), nullable=True)
    display = Column(String(255), nullable=True)
    ports = Column(Text, nullable=True)  # JSON string
    os = Column(String(100), nullable=True)
    known_issues = Column(Text, nullable=True)  # JSON string
    # Generated images are returned as base64 data URIs, which are far larger
    # than VARCHAR(500). Storing them in a narrow column raised
    # "Data too long for column 'image_url'" on MySQL and the commit that
    # persisted the image silently failed.
    image_url = Column(Text, nullable=True)
    source_urls = Column(Text, nullable=True)  # JSON string of source citations
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DiagnosticSession(Base):
    __tablename__ = 'diagnostic_sessions'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    laptop_setup_id = Column(Integer, ForeignKey("laptop_setup.id"), nullable=True)
    issue_description = Column(Text, nullable=False)
    status = Column(String(50), default="active")  # active, resolved, technician_required
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("diagnostic_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON string for additional data
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VideoTutorial(Base):
    __tablename__ = 'video_tutorials'

    id = Column(Integer, primary_key=True, index=True)
    laptop_setup_id = Column(Integer, ForeignKey("laptop_setup.id"), nullable=False)
    video_id = Column(String(100), nullable=False)  # YouTube video ID
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    video_url = Column(String(500), nullable=False)
    issue_keyword = Column(String(100), nullable=True)
    cached_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)


class Notification(Base):
    __tablename__ = 'notifications'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)  # maintenance, alert, system
    priority = Column(String(20), nullable=False)  # high, medium, low
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON string for additional data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
