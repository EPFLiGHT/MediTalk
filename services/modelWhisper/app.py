from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import whisper
import tempfile
import os
import logging
from pydantic import BaseModel
from typing import Optional
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediTalk Whisper ASR", version="1.0.0")

# Global model variable
whisper_model = None

# Debug mode - save audio files for debugging
DEBUG_AUDIO = os.getenv('DEBUG_WHISPER_AUDIO', 'true').lower() == 'true'
DEBUG_DIR = "../../outputs/whisper_debug"

if DEBUG_AUDIO:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    logger.info(f"Debug mode enabled - audio files will be saved to {DEBUG_DIR}")

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    detected_language: Optional[str] = None
    confidence: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    global whisper_model
    try:
        model_size = os.getenv('WHISPER_MODEL', 'base')  # tiny, base, small, medium, large
        logger.info(f"Loading OpenAI Whisper model: {model_size}")
        
        # Load Whisper model
        whisper_model = whisper.load_model(model_size)
        logger.info(f"OpenAI Whisper {model_size} model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        whisper_model = None

@app.get("/health")
def health_check():
    if whisper_model is None:
        return {
            "status": "unhealthy", 
            "model": "openai-whisper", 
            "error": "Model not loaded"
        }
    return {
        "status": "healthy", 
        "model": "openai-whisper",
        "supported_languages": ["auto", "en", "fr"]  # Auto-detect, English, French
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Transcribe uploaded audio file to text
    
    Args:
        audio_file: Audio file to transcribe
        language: Language code ("en", "fr", or None for auto-detection)
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    debug_file_path = None
    
    try:
        logger.info(f"Transcribing audio file: {audio_file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.info(f"Temporary file created at: {tmp_file_path}")
        
        # If debug mode, save a copy for inspection
        if DEBUG_AUDIO:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"whisper_input_{timestamp}.wav"
            debug_file_path = os.path.join(DEBUG_DIR, debug_filename)
            shutil.copy(tmp_file_path, debug_file_path)
            logger.info(f"Debug: Audio saved to {debug_file_path}")
        
        try:
            # Transcribe with Whisper using the loaded model
            # Determine language mode
            if language and language.lower() != "auto":
                logger.info(f"Starting transcription with specified language: {language}")
                transcribe_kwargs = {
                    "language": language.lower(),
                    "fp16": False
                }
            else:
                logger.info("Starting transcription with auto language detection...")
                transcribe_kwargs = {
                    "fp16": False
                    # No language parameter = auto-detect
                }
            
            # Use the whisper model directly (already loaded at startup)
            result = whisper_model.transcribe(tmp_file_path, **transcribe_kwargs)
            
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            
            logger.info(f"Transcription completed (detected: {detected_lang}): '{text[:100]}...'")
            
            response = TranscriptionResponse(
                text=text,
                language=language if language else "auto",
                detected_language=detected_lang,
                confidence=None
            )
            
            # Add debug file path to response if available
            if debug_file_path:
                logger.info(f"You can listen to this recording at: http://localhost:8080/whisper-debug/{os.path.basename(debug_file_path)}")
            
            return response
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

class TranscribeFromPathRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None

@app.post("/transcribe_from_path", response_model=TranscriptionResponse)
async def transcribe_from_path(request: TranscribeFromPathRequest):
    """
    Transcribe audio file from file system path
    
    Args:
        request: Contains audio_path (file system path) and optional language
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    audio_path = request.audio_path
    language = request.language
    
    # Verify file exists
    if not os.path.exists(audio_path):
        logger.info(f"Audio file not found: {audio_path} from directory: {os.getcwd()}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    
    debug_file_path = None
    
    try:
        logger.info(f"Transcribing audio from path: {audio_path}")
        
        # If debug mode, save a copy for inspection
        if DEBUG_AUDIO:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"whisper_input_{timestamp}.wav"
            debug_file_path = os.path.join(DEBUG_DIR, debug_filename)
            shutil.copy(audio_path, debug_file_path)
            logger.info(f"Debug: Audio saved to {debug_file_path}")
        
        # Transcribe with Whisper using the loaded model
        # Determine language mode
        if language and language.lower() != "auto":
            logger.info(f"Starting transcription with specified language: {language}")
            transcribe_kwargs = {
                "language": language.lower(),
                "fp16": False
            }
        else:
            logger.info("Starting transcription with auto language detection...")
            transcribe_kwargs = {
                "fp16": False
            }
        
        result = whisper_model.transcribe(audio_path, **transcribe_kwargs)
        
        text = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        
        logger.info(f"Transcription completed (detected: {detected_lang}): '{text[:100]}...'")
        
        response = TranscriptionResponse(
            text=text,
            language=language if language else "auto",
            detected_language=detected_lang,
            confidence=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/debug-audio/{filename}")
async def get_debug_audio(filename: str):
    """Serve debug audio files for inspection"""
    if not DEBUG_AUDIO:
        raise HTTPException(status_code=404, detail="Debug mode is disabled")
    
    file_path = os.path.join(DEBUG_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.get("/debug-audio")
async def list_debug_audio():
    """List all available debug audio files"""
    if not DEBUG_AUDIO:
        return {"debug_enabled": False, "files": []}
    
    if not os.path.exists(DEBUG_DIR):
        return {"debug_enabled": True, "files": []}
    
    files = [f for f in os.listdir(DEBUG_DIR) if f.endswith('.wav')]
    files.sort(reverse=True)  # Most recent first
    
    return {
        "debug_enabled": True,
        "files": files,
        "count": len(files),
        "directory": DEBUG_DIR
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5007)