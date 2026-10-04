#fetch.py
import json

from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any

class UserBase(BaseModel):
    email: EmailStr
    name: str
    surname: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    needs_setup: Optional[bool] = None

class TokenData(BaseModel):
    email: Optional[str] = None


class LaptopSetupBase(BaseModel):
    brand: str
    model: str


class LaptopSetupCreate(LaptopSetupBase):
    pass


class LaptopSetupResponse(LaptopSetupBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SetupStatusResponse(BaseModel):
    needs_setup: bool
    laptop: Optional[LaptopSetupResponse] = None


# Phase 1: RAG Engine Models
class LaptopSpecsBase(BaseModel):
    cpu: Optional[str] = None
    gpu: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    display: Optional[str] = None
    ports: Optional[List[str]] = None
    os: Optional[str] = None
    known_issues: Optional[List[str]] = None
    image_url: Optional[str] = None
    source_urls: Optional[List[str]] = None

    @field_validator("ports", "known_issues", "source_urls", mode="before")
    @classmethod
    def parse_json_lists(cls, value):
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        return value


class LaptopSpecsCreate(LaptopSpecsBase):
    laptop_setup_id: int


class LaptopSpecsResponse(LaptopSpecsBase):
    id: int
    laptop_setup_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LaptopIngestRequest(BaseModel):
    brand: str
    model_name: str


class LaptopIngestResponse(BaseModel):
    message: str
    laptop_setup_id: int
    specs: Optional[LaptopSpecsResponse] = None
    sources_found: int


class Laptop3DModelResponse(BaseModel):
    laptop_id: int
    brand: str
    model_name: str
    category: str
    model_url: str
    component_nodes: Dict[str, str]
    image_url: Optional[str] = None


class ExplainComponentRequest(BaseModel):
    component_name: str
    laptop_brand: str
    laptop_model: str


class ExplainComponentResponse(BaseModel):
    component_name: str
    laptop_brand: str
    laptop_model: str
    explanation: str


# Phase 2: Chat & Safety Models
class ChatMessageRequest(BaseModel):
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    message: str


class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    response_text: str
    is_high_risk: bool
    warning_data: Optional[Dict[str, Any]] = None
    session_id: int


class ChatHistoryResponse(BaseModel):
    session_id: int
    messages: List[ChatMessageResponse]


# Phase 3: YouTube & Notifications Models
class VideoTutorialResponse(BaseModel):
    video_id: str
    title: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    channel_title: str = "YouTube"
    url: str
    cached_at: Optional[str] = None


class NotificationResponse(BaseModel):
    id: int
    title: str
    message: str
    notification_type: str
    priority: str
    is_read: bool
    created_at: str

    class Config:
        from_attributes = True
