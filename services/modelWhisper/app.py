from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import logging
from pydantic import BaseModel
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediTalk Whisper ASR", version="1.0.0")

# Global model variable
whisper_model = None

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
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
        return {"status": "unhealthy", "model": "openai-whisper", "error": "Model not loaded"}
    return {"status": "healthy", "model": "openai-whisper"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file to text"""
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    try:
        logger.info(f"Transcribing audio file: {audio_file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.info(f"Temporary file created at: {tmp_file_path}")
        
        try:
            # Transcribe with Whisper using the loaded model
            logger.info("Starting transcription process using loaded Whisper model...")
            
            # Use the whisper model directly (already loaded at startup)
            result = whisper_model.transcribe(
                tmp_file_path,
                language="en",
                fp16=False
            )
            
            text = result["text"].strip()
            logger.info(f"Transcription completed: '{text[:100]}...'")
            
            # Get detected language (for info)
            detected_language = result.get("language", "en")
            
            return TranscriptionResponse(
                text=text,
                language=detected_language,
                confidence=None
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-stream")
async def transcribe_stream():
    """Future endpoint for real-time streaming transcription"""
    # TODO: Implement streaming transcription
    return {"message": "Streaming transcription not yet implemented"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5007)