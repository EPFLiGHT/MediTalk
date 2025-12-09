from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ============================================================================
# Data Models for Conversation Management
# ============================================================================

class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageContent(BaseModel):
    """
    Generic message content. Can be text or audio path
    Service-specific parsing happens in the service itself.
    """
    type: str  # "text", "audio"
    data: Union[str, Dict[str, Any]]  # Path or structured data
    metadata: Optional[Dict[str, Any]] = None  # Service-specific metadata


class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: List[MessageContent]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_metadata: Optional[Dict[str, Any]] = None  # Which service generated this


class Conversation(BaseModel):
    """Conversation object stored by controller"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[ConversationMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None  # Session info, user prefs, etc.


# ============================================================================
# Controller API Request/Response Models
# ============================================================================


class ChatRequest(BaseModel):
    """
    Request to the controller's chat endpoint.
    Controller forwards to appropriate services based on content types.
    """
    conversation_id: Optional[str] = None  # if None, creates new conversation
    message: ConversationMessage  # User's message
    
    # which services to call
    use_stt: bool = True  # Transcribe audio with Whisper?
    use_llm: bool = True  # Generate response with MultiMeditron?
    use_tts: bool = True  # Synthesize audio response?
    
    llm_service: str = "multimeditron"  # only "multimeditron"
    tts_service: str = "orpheus"  # "orpheus" | "bark" | "csm" | "qwen3omni"
    tts_language: Optional[str] = "en"  # "en" only, "fr" only for orpheus for now
    
    # Additional context
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response from controller's chat endpoint"""
    conversation_id: str
    assistant_message: ConversationMessage  # Generated response
    processing_info: Dict[str, Any]  # Which services were called, timing, ...


class TranscribeRequest(BaseModel):
    """Direct transcription request (bypasses conversation storage)"""
    audio_path: str
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Transcription result"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    service_metadata: Optional[Dict[str, Any]] = None


class SynthesizeRequest(BaseModel):
    """Direct TTS request (bypasses conversation storage)"""
    text: str
    language: str = "en"
    service: str = "qwen3omni"  # "orpheus" | "bark" | "csm" | "qwen3omni"
    voice_params: Optional[Dict[str, Any]] = None  # Service-specific params


class SynthesizeResponse(BaseModel):
    """Synthesis result"""
    audio_path: str  # Where generated audio was saved
    duration: Optional[float] = None
    service_metadata: Optional[Dict[str, Any]] = None


class ConversationListResponse(BaseModel):
    """List of conversations"""
    conversations: List[Conversation]
    total: int


class ServiceStatus(BaseModel):
    """Health status of a single service"""
    name: str
    url: str
    status: str  # "healthy" | "unhealthy" | "unknown"
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall system health"""
    controller_status: str
    services: List[ServiceStatus]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
