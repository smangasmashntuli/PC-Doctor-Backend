#main.py
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from typing import List
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)

from backend.app.safety_guardrails import RiskLevel

from backend.app.database import engine, get_db, SessionLocal
from backend.app.models import Base, Users as User, LaptopSetup, LaptopSpecs, DiagnosticSession, ChatMessage, VideoTutorial, Notification
from backend.app.fetch import (
    UserCreate,
    UserResponse,
    UserLogin,
    Token,
    LaptopSetupCreate,
    LaptopSetupResponse,
    SetupStatusResponse,
)
from backend.app.crud import get_user_by_email, get_user_by_name, create_user
from backend.app.auth import authenticate_user, create_access_token, get_current_user
from backend.app.database import sync_users_table_schema
from backend.app.setup import create_laptop_setup, get_laptop_setup_by_user_id
from backend.app.rag_engine import get_rag_engine
from backend.app.gemini_service import get_gemini_service
from backend.app.safety_guardrails import get_safety_guardrails
from backend.app.fetch import (
    LaptopIngestRequest,
    LaptopIngestResponse,
    ChatMessageRequest,
    ChatResponse,
    ChatHistoryResponse,
    VideoTutorialResponse,
    NotificationResponse,
    Laptop3DModelResponse,
    ExplainComponentRequest,
    ExplainComponentResponse,
)
from backend.app.youtube_service import get_youtube_service
from backend.app.notifications import get_notification_service

sync_users_table_schema()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Native API", description="Authentication API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:19006",
        "http://localhost:19000",
        "http://localhost:8081",
        "http://127.0.0.1:19006",
        "exp://127.0.0.1:19000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMPONENT_NODE_MAPPING = {
    "RAM": "mesh_ram_01",
    "SSD": "mesh_nvme_01",
    "Battery": "mesh_batt_01",
    "Fan": "mesh_fan_01",
    "Cover": "mesh_bottom_cover",
}

LAPTOP_3D_MODEL_URLS = {
    "gaming": "https://assets.pc-docter-ai.local/models/gaming-laptop.glb",
    "thin_light": "https://assets.pc-docter-ai.local/models/thin-light-laptop.glb",
    "default": "https://assets.pc-docter-ai.local/models/default-laptop.glb",
}


def _infer_laptop_category(brand: str, model_name: str, specs: LaptopSpecs | None = None) -> str:
    combined_text = f"{brand} {model_name}".lower()
    gpu_text = (specs.gpu if specs and specs.gpu else "").lower()

    gaming_keywords = [
        "gaming",
        "alienware",
        "legion",
        "rog",
        "zephyrus",
        "predator",
        "omen",
        "victus",
        "nitro",
        "blade",
        "rtx",
        "gtx",
    ]
    thin_light_keywords = [
        "xps",
        "thinkpad x1",
        "thinkpad t",
        "macbook",
        "zenbook",
        "spectre",
        "swift",
        "yoga slim",
        "surface laptop",
        "gram",
    ]

    if any(keyword in combined_text for keyword in gaming_keywords) or any(keyword in gpu_text for keyword in ["rtx", "gtx", "radeon rx"]):
        return "gaming"

    if any(keyword in combined_text for keyword in thin_light_keywords) and not any(keyword in gpu_text for keyword in ["rtx", "gtx", "radeon rx"]):
        return "thin_light"

    return "default"


def _get_laptop_3d_model_url(category: str) -> str:
    return LAPTOP_3D_MODEL_URLS.get(category, LAPTOP_3D_MODEL_URLS["default"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Native API"}

@app.get("/ping")
@app.get("/ping/")
def ping():
    return {"status": "ok"}

@app.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@app.post("/signup/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user_email = get_user_by_email(db, email=user.email)
    if db_user_email:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_user = create_user(db=db, user=user)
    return new_user


@app.post("/login", response_model=Token)
@app.post("/login/", response_model=Token)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, email=user.email, password=user.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    access_token_expires = timedelta(minutes=50)
    has_setup = get_laptop_setup_by_user_id(db, user.id) is not None
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "needs_setup": not has_setup}

@app.get("/me", response_model=UserResponse)
@app.get("/me/", response_model=UserResponse)
def read_user_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/setup", response_model=LaptopSetupResponse, status_code=status.HTTP_201_CREATED)
@app.post("/setup/", response_model=LaptopSetupResponse, status_code=status.HTTP_201_CREATED)
async def register_laptop(
    laptop: LaptopSetupCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Create laptop setup
    laptop_setup = create_laptop_setup(db=db, user=current_user, laptop=laptop)
    
    # Trigger RAG ingestion in background (don't block the response)
    try:
        rag_engine = get_rag_engine()
        background_tasks.add_task(
            rag_engine.ingest_laptop_specs,
            brand=laptop.brand,
            model_name=laptop.model,
            user_id=current_user.id
        )
    except Exception as e:
        # Log error but don't fail the setup
        print(f"Warning: Failed to trigger RAG ingestion: {e}")
    
    return laptop_setup


@app.get("/setup/status", response_model=SetupStatusResponse)
@app.get("/setup/status/", response_model=SetupStatusResponse)
def setup_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    laptop = get_laptop_setup_by_user_id(db, current_user.id)
    return {
        "needs_setup": laptop is None,
        "laptop": laptop,
    }


@app.post("/laptop/specs/ingest", response_model=LaptopIngestResponse)
@app.post("/laptop/specs/ingest/", response_model=LaptopIngestResponse)
async def ingest_laptop_specs(
    request: LaptopIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingest laptop specifications from web sources and store in vector database
    
    - **brand**: Laptop brand (e.g., "Dell", "Lenovo", "HP")
    - **model_name**: Model name/number (e.g., "XPS 15 9520", "ThinkPad T14")
    """
    try:
        rag_engine = get_rag_engine()
        result = await rag_engine.ingest_laptop_specs(
            brand=request.brand,
            model_name=request.model_name,
            user_id=current_user.id
        )

        try:
            gemini = get_gemini_service()
            image_url = await gemini.generate_laptop_image(
                brand=request.brand,
                model_name=request.model_name,
            )

            specs_record = db.query(LaptopSpecs).filter(
                LaptopSpecs.laptop_setup_id == result["laptop_setup_id"]
            ).first()
            if specs_record:
                specs_record.image_url = image_url
                specs_record.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(specs_record)

                if result.get("specs"):
                    result["specs"].image_url = image_url
        except Exception as image_error:
            logger.warning(f"Laptop image generation skipped: {image_error}")
        
        return LaptopIngestResponse(
            message=result["message"],
            laptop_setup_id=result["laptop_setup_id"],
            specs=result.get("specs"),
            sources_found=result["sources_found"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest laptop specs: {str(e)}"
        )


@app.get("/laptop/{laptop_id}/3d-model", response_model=Laptop3DModelResponse)
@app.get("/laptop/{laptop_id}/3d-model/", response_model=Laptop3DModelResponse)
def get_laptop_3d_model(
    laptop_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    laptop = db.query(LaptopSetup).filter(
        LaptopSetup.id == laptop_id,
        LaptopSetup.user_id == current_user.id
    ).first()

    if not laptop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Laptop not found"
        )

    specs = db.query(LaptopSpecs).filter(
        LaptopSpecs.laptop_setup_id == laptop.id
    ).first()

    category = _infer_laptop_category(laptop.brand, laptop.model, specs)
    return Laptop3DModelResponse(
        laptop_id=laptop.id,
        brand=laptop.brand,
        model_name=laptop.model,
        category=category,
        model_url=_get_laptop_3d_model_url(category),
        component_nodes=COMPONENT_NODE_MAPPING,
        image_url=specs.image_url if specs else None,
    )


@app.post("/chat/explain-component", response_model=ExplainComponentResponse)
@app.post("/chat/explain-component/", response_model=ExplainComponentResponse)
async def explain_component(
    request: ExplainComponentRequest,
    current_user: User = Depends(get_current_user),
    gemini = Depends(get_gemini_service),
):
    try:
        explanation = await gemini.explain_component(
            component_name=request.component_name,
            laptop_brand=request.laptop_brand,
            laptop_model=request.laptop_model,
        )

        return ExplainComponentResponse(
            component_name=request.component_name,
            laptop_brand=request.laptop_brand,
            laptop_model=request.laptop_model,
            explanation=explanation,
        )
    except Exception as e:
        logger.error(f"Error explaining component: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to explain component"
        )


@app.post("/chat/message", response_model=ChatResponse)
@app.post("/chat/message/", response_model=ChatResponse)
async def chat_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    gemini = Depends(get_gemini_service),
    safety = Depends(get_safety_guardrails)
):
    """
    Send a chat message to the AI troubleshooting assistant
    
    - **user_id**: User ID (from authentication)
    - **session_id**: Optional session ID for continuing conversations
    - **message**: User's question or issue description
    """
    try:
        # Get or create diagnostic session
        if request.session_id:
            session = db.query(DiagnosticSession).filter(
                DiagnosticSession.id == request.session_id,
                DiagnosticSession.user_id == current_user.id
            ).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found"
                )
        else:
            # Create new session
            session = DiagnosticSession(
                user_id=current_user.id,
                laptop_setup_id=None,  # Will be set if user has laptop
                issue_description=request.message,
                status="active"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            request.session_id = session.id
        
        # Save user message
        user_message = ChatMessage(
            session_id=session.id,
            role="user",
            content=request.message
        )
        db.add(user_message)
        db.commit()
        
        # Check safety guardrails (already injected via dependency)
        risk_level, warning_data = safety.assess_risk(request.message)
        
        if risk_level == RiskLevel.HIGH:
            # High-risk issue detected - return warning
            response_text = warning_data['warning_message'] + "\n\n" + warning_data['action_recommendation']
            
            # Save assistant response
            assistant_message = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=response_text,
                metadata_json=json.dumps({"is_high_risk": True, "category": warning_data.get('category')})
            )
            db.add(assistant_message)
            db.commit()
            
            return ChatResponse(
                response_text=response_text,
                is_high_risk=True,
                warning_data=warning_data,
                session_id=session.id
            )
        
        # Generate AI response (already injected via dependency)
        response_text = await gemini.get_chat_response(
            user_query=request.message,
            user_id=current_user.id,
            session_id=session.id
        )
        
        response_text = response_text.get('response_text', 'I apologize, but I encountered an error. Please try again.')
        
        # Save assistant response
        assistant_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response_text,
            metadata_json=json.dumps({"is_high_risk": False})
        )
        db.add(assistant_message)
        db.commit()
        
        return ChatResponse(
            response_text=response_text,
            is_high_risk=False,
            warning_data=None,
            session_id=session.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@app.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
@app.get("/chat/history/{session_id}/", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get chat history for a specific session
    
    - **session_id**: The diagnostic session ID
    """
    # Verify session belongs to user
    session = db.query(DiagnosticSession).filter(
        DiagnosticSession.id == session_id,
        DiagnosticSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get messages
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages
    )


# Phase 3: YouTube & Notifications Endpoints

@app.get("/videos/search", response_model=List[VideoTutorialResponse])
@app.get("/videos/search/", response_model=List[VideoTutorialResponse])
async def search_youtube_tutorials(
    brand: str,
    model: str,
    issue: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    youtube = Depends(get_youtube_service)
):
    """
    Search for YouTube tutorial videos with caching
    
    - **brand**: Laptop brand (e.g., "Dell", "Lenovo")
    - **model**: Laptop model (e.g., "XPS 15 9520")
    - **issue**: Issue keyword (e.g., "overheating", "won't boot")
    
    Returns cached results if available (24-hour TTL), otherwise fetches from YouTube API
    """
    try:
        videos = await youtube.search_youtube_tutorials(
            brand=brand,
            model=model,
            issue_keyword=issue,
            user_id=current_user.id,
            db=db
        )
        
        return videos
        
    except Exception as e:
        logger.error(f"Error searching YouTube videos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search YouTube videos"
        )


@app.get("/notifications", response_model=List[NotificationResponse])
@app.get("/notifications/", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    notification_service = Depends(get_notification_service)
):
    """
    Get notifications for the current user
    
    - **unread_only**: If true, only return unread notifications (default: false)
    """
    try:
        notifications = notification_service.get_user_notifications(
            user_id=current_user.id,
            db=db,
            unread_only=unread_only
        )
        
        return notifications
        
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch notifications"
        )


@app.patch("/notifications/{notification_id}/read", response_model=NotificationResponse)
@app.patch("/notifications/{notification_id}/read/", response_model=NotificationResponse)
async def mark_notification_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    notification_service = Depends(get_notification_service)
):
    """
    Mark a notification as read
    
    - **notification_id**: The notification ID to mark as read
    """
    try:
        notification = notification_service.mark_notification_as_read(
            notification_id=notification_id,
            user_id=current_user.id,
            db=db
        )
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notification as read"
        )


@app.post("/notifications/trigger-maintenance")
@app.post("/notifications/trigger-maintenance/")
async def trigger_maintenance_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    notification_service = Depends(get_notification_service)
):
    """
    Trigger maintenance alert generation for the current user
    
    Creates personalized maintenance reminders based on the user's laptop specs
    """
    try:
        notifications = notification_service.generate_maintenance_alerts(
            user_id=current_user.id,
            db=db
        )
        
        return {
            "message": f"Generated {len(notifications)} maintenance alerts",
            "notifications": notifications
        }
        
    except Exception as e:
        logger.error(f"Error generating maintenance alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate maintenance alerts"
        )




"""@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=engine)

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "API is running"}"""