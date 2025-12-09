"""
Controller Service - Orchestrates model services and manages conversations.

Philosophy:
- Thin orchestration layer - no business logic
- Services are autonomous - controller just coordinates
- Stateful (conversation storage) but horizontally scalable
- Configuration-driven service discovery

Architecture:
    Streamlit → Controller → [MultiMeditron, Whisper, Orpheus, ...]
                    ↓
              [Conversation
                 Storage]

Endpoints:
- POST /chat              - Full conversation pipeline
- POST /transcribe        - Direct STT
- POST /synthesize        - Direct TTS
- GET  /conversations     - List conversations
- GET  /conversations/{id} - Get specific conversation
- DELETE /conversations/{id} - Delete conversation
- GET  /health            - System health check
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .conversation_manager import ConversationManager
from .models import (
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationListResponse,
    ConversationMessage,
    HealthResponse,
    MessageContent,
    MessageRole,
    ServiceStatus,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from .service_clients import ServiceRegistry, call_llm, call_stt, call_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Service Configuration (from environment variables)
# ============================================================================

SERVICE_CONFIG = {
    "multimeditron": os.getenv("MULTIMEDITRON_URL", "http://localhost:5000"),
    "whisper": os.getenv("WHISPER_URL", "http://localhost:5007"),
    "orpheus": os.getenv("ORPHEUS_URL", "http://localhost:5005"),
    "bark": os.getenv("BARK_URL", "http://localhost:5006"),
    "csm": os.getenv("CSM_URL", "http://localhost:5010"),
    "qwen3omni": os.getenv("QWEN3OMNI_URL", "http://localhost:5009"),
}

# Conversation storage directory (shared with services)
CONVERSATION_STORAGE_DIR = os.getenv(
    "CONVERSATION_STORAGE_DIR",
    "../../inputs/controller/conversations"  # Shared storage location
)

# Verbose logging
VERBOSE = os.getenv("CONTROLLER_VERBOSE", "false").lower() == "true"

# Global service registry and conversation manager
service_registry: ServiceRegistry = None
conversation_manager: ConversationManager = None


# ============================================================================
# FastAPI Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global service_registry, conversation_manager
    
    # Startup
    logger.info("🚀 Starting Controller Service...")
    logger.info(f"Service configuration: {SERVICE_CONFIG}")
    logger.info(f"Conversation storage: {CONVERSATION_STORAGE_DIR}")
    logger.info(f"Verbose logging: {VERBOSE}")
    
    # Initialize conversation manager with JSON file storage
    conversation_manager = ConversationManager(
        storage_dir=CONVERSATION_STORAGE_DIR,
        verbose=VERBOSE
    )
    
    service_registry = ServiceRegistry(SERVICE_CONFIG, verbose=VERBOSE)
    
    # Check service health
    health = await service_registry.health_check_all()
    for name, status in health.items():
        status_emoji = "✅" if status["status"] == "healthy" else "❌"
        logger.info(f"{status_emoji} {name}: {status['status']}")
    
    logger.info("✅ Controller Service ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Controller Service...")
    await service_registry.close_all()
    logger.info("✅ Controller Service stopped")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="MediTalk Controller",
    description="Orchestration service for medical conversation AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of controller and all model services.
    
    Returns detailed status for each service.
    """
    service_health = await service_registry.health_check_all()
    
    service_statuses = [
        ServiceStatus(
            name=name,
            url=SERVICE_CONFIG[name],
            status=health.get("status", "unknown"),
            response_time_ms=health.get("response_time_ms"),
            error=health.get("error")
        )
        for name, health in service_health.items()
    ]
    
    return HealthResponse(
        controller_status="healthy",
        services=service_statuses
    )


@app.get("/stats")
async def get_stats():
    """Get controller statistics"""
    return {
        "conversation_stats": conversation_manager.get_stats(),
        "service_count": len(service_registry.clients)
    }


# ============================================================================
# Conversation Management Endpoints
# ============================================================================

@app.post("/conversations", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(metadata: Dict = None):
    """Create a new conversation"""
    conversation = conversation_manager.create_conversation(metadata=metadata)
    logger.info(f"Created conversation {conversation.id}")
    return conversation


@app.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(limit: int = 100):
    """List all conversations"""
    conversations = conversation_manager.list_conversations(limit=limit)
    return ConversationListResponse(
        conversations=conversations,
        total=len(conversations)
    )


@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    return conversation


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    deleted = conversation_manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )


# ============================================================================
# Direct Service Endpoints (Bypass conversation storage)
# ============================================================================

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    Direct transcription without conversation storage.
    
    Forwards request to Whisper service.
    """
    whisper_client = service_registry.get("whisper")
    if not whisper_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper service not available"
        )
    
    try:
        result = await call_stt(
            whisper_client,
            request.audio_path,
            language=request.language
        )
        
        return TranscribeResponse(
            text=result.get("text", ""),
            language=result.get("language"),
            confidence=result.get("confidence"),
            service_metadata=result
        )
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Direct TTS without conversation storage.
    
    Forwards request to specified TTS service (Orpheus, Bark, CSM).
    """
    tts_client = service_registry.get(request.service)
    if not tts_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS service '{request.service}' not available"
        )
    
    try:
        result = await call_tts(
            tts_client,
            request.text,
            language=request.language,
            **(request.voice_params or {})
        )
        
        return SynthesizeResponse(
            audio_path=result.get("audio_path", ""),
            duration=result.get("duration"),
            service_metadata=result
        )
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}"
        )


# ============================================================================
# Main Chat Endpoint (Full Pipeline Orchestration)
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - orchestrates full conversation pipeline.
    
    Pipeline:
    1. Create/get conversation
    2. If audio in message → STT (Whisper)
    3. Call LLM with conversation history
    4. If TTS enabled → synthesize response
    5. Store all messages in conversation
    6. Return assistant response
    
    Philosophy: Controller orchestrates, services execute.
    Each service owns its logic - controller just connects them.
    """
    
    # Step 1: Get or create conversation
    if request.conversation_id:
        conversation = conversation_manager.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {request.conversation_id} not found"
            )
    else:
        conversation = conversation_manager.create_conversation()
    
    conversation_id = conversation.id
    processing_info = {"services_called": []}
    
    try:
        # Step 2: Process audio input with STT (if present and enabled)
        user_message = request.message
        
        if VERBOSE:
            logger.info(f"Processing chat for conversation {conversation_id}")
            logger.info(f"User message content types: {[c.type for c in user_message.content]}")
        
        if request.use_stt:
            # Check if message contains audio
            audio_content = next(
                (c for c in user_message.content if c.type == "audio"),
                None
            )
            
            if audio_content:
                whisper_client = service_registry.get("whisper")
                if whisper_client:
                    if VERBOSE:
                        logger.info(f"Transcribing audio from {audio_content.data}")
                    
                    stt_result = await call_stt(
                        whisper_client,
                        audio_content.data,
                        verbose=VERBOSE
                    )
                    
                    # Add transcribed text to message
                    user_message.content.append(
                        MessageContent(
                            type="text",
                            data=stt_result.get("text", ""),
                            metadata={"source": "whisper"}
                        )
                    )
                    processing_info["services_called"].append("whisper")
                    processing_info["stt_result"] = stt_result
        
        # Store user message (saves to JSON file)
        conversation_manager.add_message(conversation_id, user_message, verbose=VERBOSE)
        
        # Step 3: Generate LLM response (if enabled)
        assistant_text = ""
        
        if request.use_llm:
            llm_client = service_registry.get(request.llm_service)
            if not llm_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"LLM service '{request.llm_service}' not available"
                )
            
            if VERBOSE:
                logger.info(f"Generating response with {request.llm_service}...")
            
            # Pass conversation JSON file path to service (not the data itself)
            conversation_json_path = conversation_manager.get_conversation_path(conversation_id)
            
            llm_result = await call_llm(
                llm_client,
                conversation_json_path=conversation_json_path,  # Changed: pass file path
                verbose=VERBOSE,
                **(request.context or {})
            )
            
            assistant_text = llm_result.get("response", "")
            processing_info["services_called"].append(request.llm_service)
            processing_info["llm_result"] = llm_result
        
        # Step 4: Synthesize audio response (if enabled)
        assistant_message_content = [
            MessageContent(type="text", data=assistant_text)
        ]
        
        if request.use_tts and assistant_text:
            tts_client = service_registry.get(request.tts_service)
            if tts_client:
                if VERBOSE:
                    logger.info(f"Synthesizing audio with {request.tts_service}...")
                
                tts_result = await call_tts(
                    tts_client,
                    assistant_text,
                    language=request.tts_language,
                    verbose=VERBOSE
                )
                
                assistant_message_content.append(
                    MessageContent(
                        type="audio",
                        data=tts_result.get("audio_path", ""),
                        metadata={"source": request.tts_service}
                    )
                )
                processing_info["services_called"].append(request.tts_service)
                processing_info["tts_result"] = tts_result
        
        # Step 5: Create and store assistant message
        assistant_message = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=assistant_message_content,
            service_metadata={
                "llm_service": request.llm_service,
                "tts_service": request.tts_service if request.use_tts else None
            }
        )
        
        conversation_manager.add_message(conversation_id, assistant_message, verbose=VERBOSE)
        
        # Step 6: Return response
        if VERBOSE:
            logger.info(f"Chat completed for conversation {conversation_id}")
            logger.info(f"Services called: {processing_info['services_called']}")
        
        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            processing_info=processing_info
        )
    
    except Exception as e:
        logger.error(f"Chat pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline failed: {str(e)}"
        )


# ============================================================================
# Run Server (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("CONTROLLER_PORT", "8000"))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
