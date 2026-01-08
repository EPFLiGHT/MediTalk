import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List
import uvicorn

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from conversation_manager import ConversationManager
from data_models import (
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
    TranscribeWithConversationRequest,
    TranscribeWithConversationResponse,
)
from service_clients import ServiceRegistry, call_multimeditron, call_stt, call_tts

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
    "multimeditron": os.getenv("MULTIMEDITRON_URL", "http://localhost:5009"),
    "whisper": os.getenv("WHISPER_URL", "http://localhost:5007"),
    "orpheus": os.getenv("ORPHEUS_URL", "http://localhost:5005"),
    "bark": os.getenv("BARK_URL", "http://localhost:5008"),
    "csm": os.getenv("CSM_URL", "http://localhost:5010"),
    "qwen3omni": os.getenv("QWEN3OMNI_URL", "http://localhost:5014"),
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
conversation_manager: ConversationManager = None


# ============================================================================
# FastAPI Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global service_registry, conversation_manager
    
    # Startup
    logger.info("Starting Controller Service...")
    logger.info(f"Service configuration: {SERVICE_CONFIG}")
    logger.info(f"Conversation storage: {CONVERSATION_STORAGE_DIR}")
    logger.info(f"Verbose logging: {VERBOSE}")
    
    # Initialize conversation manager with JSON file storage
    conversation_manager = ConversationManager(
        storage_dir=CONVERSATION_STORAGE_DIR,
        verbose=VERBOSE
    )
    
    service_registry = ServiceRegistry(SERVICE_CONFIG, verbose=VERBOSE)
    
    # Check service health (non-blocking - don't wait if services are slow)
    try:
        health = await service_registry.health_check_all()
        for name, status in health.items():
            status_symbol = "[OK]" if status["status"] == "healthy" else "[FAIL]"
            logger.info(f"{status_symbol} {name}: {status['status']}")
    except Exception as e:
        logger.warning(f"Health check failed during startup: {e}")
        logger.info("Controller will start anyway - services can be checked later via /health endpoint")
    
    logger.info("Controller Service ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Controller Service...")
    await service_registry.close_all()
    logger.info("Controller Service stopped")


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
    allow_origins=["*"],
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


@app.post("/transcribe_with_conversation", response_model=TranscribeWithConversationResponse)
async def transcribe_with_conversation(request: TranscribeWithConversationRequest):
    """Transcribe audio and save to conversation."""
    # Get or create conversation
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
    
    # Call Whisper service
    whisper_client = service_registry.get("whisper")
    if not whisper_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper service not available"
        )
    
    try:
        logger.info(f"Transcribing audio for conversation {conversation_id}: {request.audio_path}")
        
        result = await call_stt(
            whisper_client,
            request.audio_path,
            language=request.language,
            verbose=VERBOSE
        )
        
        text = result.get("text", "")
        detected_lang = result.get("detected_language", "unknown")
        
        logger.info(f"Transcription completed (detected: {detected_lang}): '{text[:50]}...'")
        
        # Check if language is supported (English or French only)
        if detected_lang.lower() not in ['en', 'fr', 'english', 'french']:
            logger.warning(f"Unsupported language detected: {detected_lang}")
            return TranscribeWithConversationResponse(
                conversation_id=conversation_id,
                text="",
                language=request.language,
                detected_language=detected_lang,
                success=False,
                error=f"Language not supported: {detected_lang}. Please use English or French only."
            )
        
        # Save to conversation (audio path + transcription text)
        # Normalize language
        normalized_lang = 'en' if detected_lang.lower() in ['en', 'english'] else 'fr'
        
        user_message = ConversationMessage(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type="audio",
                    data=request.audio_path,
                    metadata={"source": "user_recording"}
                ),
                MessageContent(
                    type="text",
                    data=text,
                    metadata={"source": "whisper", "detected_language": normalized_lang}
                )
            ],
            service_metadata={"stt_service": "whisper"}
        )
        
        conversation_manager.add_message(conversation_id, user_message, verbose=VERBOSE)
        
        logger.info(f"Saved transcription to conversation {conversation_id}")
        
        return TranscribeWithConversationResponse(
            conversation_id=conversation_id,
            text=text,
            language=request.language,
            detected_language=normalized_lang,
            success=True,
            error=None
        )
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # Don't save anything on error
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


@app.get("/outputs/{service}/{filename}")
async def serve_audio_file(service: str, filename: str):
    """
    Serve audio files from TTS service output directories.
    This allows the Streamlit UI to load audio without the FastAPI webui.
    """
    logger.info(f"Serving audio file: /outputs/{service}/{filename}")
    
    # Get project root (controller is in services/controller)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    file_path = os.path.join(project_root, "outputs", service, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: /outputs/{service}/{filename}"
        )
    
    logger.info(f"Serving file: {file_path}")
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600"
        }
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
    
    # Initialize timing tracking
    timings = {}
    pipeline_start = time.time()
    
    # Log incoming request
    logger.info(f"Received /chat request for conversation {conversation_id}")
    logger.info(f"   - use_llm: {request.use_llm}, llm_service: {request.llm_service}")
    logger.info(f"   - use_tts: {request.use_tts}, tts_service: {request.tts_service}")
    logger.info(f"   - use_stt: {request.use_stt}")
    
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
                    logger.info(f"Calling Whisper STT for audio: {audio_content.data}")
                    
                    stt_start = time.time()
                    stt_result = await call_stt(
                        whisper_client,
                        audio_content.data,
                        verbose=VERBOSE
                    )
                    timings['stt'] = time.time() - stt_start
                    
                    logger.info(f"Whisper STT completed: '{stt_result.get('text', '')[:50]}...' (took {timings['stt']:.2f}s)")
                    
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
        
        # Store user message (saves to JSON file) --> skip if all text content is empty
        text_contents = [c.data for c in user_message.content if c.type == "text"]
        has_non_empty_text = any(text.strip() for text in text_contents if isinstance(text, str))
        
        if has_non_empty_text:
            conversation_manager.add_message(conversation_id, user_message, verbose=VERBOSE)
        else:
            logger.info(f"/!\ Skipping user message save (empty text content)")
        
        # Step 3: Generate LLM response (if enabled)
        assistant_text = ""
        
        if request.use_llm:
            llm_client = service_registry.get(request.llm_service)
            if not llm_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"LLM service '{request.llm_service}' not available"
                )
            
            # Pass conversation JSON file path to service (not the data itself)
            conversation_json_path = conversation_manager.get_conversation_path(conversation_id)
            
            logger.info(f"Calling {request.llm_service} LLM with conversation: {conversation_json_path}")
            
            llm_start = time.time()
            llm_result = await call_multimeditron(
                llm_client,
                conversation_json_path=conversation_json_path,
                verbose=VERBOSE,
                **(request.context or {})
            )
            timings['llm'] = time.time() - llm_start
            
            assistant_text = llm_result.get("response", "")
            logger.info(f"{request.llm_service} response received: '{assistant_text[:100]}...' (took {timings['llm']:.2f}s)")
            
            processing_info["services_called"].append(request.llm_service)
            processing_info["llm_result"] = llm_result
            
            # Immediately create and save assistant message with text only
            assistant_message_content = [
                MessageContent(type="text", data=assistant_text)
            ]
            
            assistant_message = ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=assistant_message_content,
                service_metadata={
                    "llm_service": request.llm_service
                }
            )
            
            # Save assistant text message to conversation JSON
            conversation_manager.add_message(conversation_id, assistant_message, verbose=VERBOSE)
            
            if VERBOSE:
                logger.info(f"Saved assistant text message to conversation {conversation_id}")
        
        # Step 4: Synthesize audio response (if enabled)
        if request.use_tts and assistant_text:
            tts_client = service_registry.get(request.tts_service)
            if tts_client:
                # Get updated conversation path (with timestamped filename)
                conversation_json_path = conversation_manager.get_conversation_path(conversation_id)
                
                logger.info(f"Calling {request.tts_service} TTS for text: '{assistant_text[:50]}...'")
                logger.info(f"   - Conversation path: {conversation_json_path}")
                logger.info(f"   - Language: {request.tts_language}")
                logger.info(f"   - Context: {request.context}")
                
                # Call TTS with conversation path (TTS extracts last assistant message)
                tts_start = time.time()
                tts_result = await call_tts(
                    tts_client,
                    target_text=assistant_text,  # For services that need it
                    conversation_json_path=conversation_json_path,
                    language=request.tts_language,
                    verbose=VERBOSE,
                    **(request.context or {})
                )
                timings['tts'] = time.time() - tts_start
                
                filename = tts_result.get("filename", "")
                service = tts_result.get("service", request.tts_service)
                logger.info(f"{request.tts_service} TTS completed: {filename} (took {timings['tts']:.2f}s)")
                
                processing_info["services_called"].append(request.tts_service)
                processing_info["tts_result"] = tts_result
                
                if filename:
                    # Construct simple URL path
                    audio_url_path = f"/outputs/{service}/{filename}"
                    logger.info(f"   Audio URL: {audio_url_path}")
                    
                    # Update the existing assistant message with audio
                    conversation = conversation_manager.get_conversation(conversation_id)
                    if conversation and conversation.messages:
                        last_message = conversation.messages[-1]
                        if last_message.role == MessageRole.ASSISTANT:
                            # Add audio to existing assistant message (use URL path, not filesystem path)
                            last_message.content.append(
                                MessageContent(
                                    type="audio",
                                    data=audio_url_path,
                                    metadata={"source": request.tts_service}
                                )
                            )
                            # Update service metadata
                            if last_message.service_metadata:
                                last_message.service_metadata["tts_service"] = request.tts_service
                            
                            # Save updated conversation
                            conversation.updated_at = datetime.utcnow()
                            conversation_manager._save_to_file(conversation, verbose=VERBOSE)
                            
                            logger.info(f"Updated assistant message with audio URL: {audio_url_path}")
                            
                            # Update assistant_message for response
                            assistant_message = last_message
        
        # Step 5: Return response
        timings['total'] = time.time() - pipeline_start
        
        logger.info(f"Chat completed for conversation {conversation_id}")
        logger.info(f"   - Services called: {processing_info['services_called']}")
        logger.info(f"Pipeline Timings:")
        if 'stt' in timings:
            logger.info(f"   - STT (Whisper): {timings['stt']:.2f}s")
        if 'llm' in timings:
            logger.info(f"   - LLM ({request.llm_service}): {timings['llm']:.2f}s")
        if 'tts' in timings:
            logger.info(f"   - TTS ({request.tts_service}): {timings['tts']:.2f}s")
        logger.info(f"   - TOTAL END-TO-END: {timings['total']:.2f}s")
        logger.info(f"   - Assistant message has {len(assistant_message.content)} content items")
        for content in assistant_message.content:
            logger.info(f"     • {content.type}: {str(content.data)[:80]}...")
        
        # Add timings to processing_info for potential client use
        processing_info['timings'] = timings
        
        response = ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            processing_info=processing_info
        )
        
        logger.info(f"Sending response to client")
        return response
    
    except Exception as e:
        logger.error(f"Chat pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline failed: {str(e)}"
        )

if __name__ == "__main__":    
    port = int(os.getenv("CONTROLLER_PORT", "8000"))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
