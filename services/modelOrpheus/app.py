from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from infer import OrpheusTTS
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orpheus TTS Service", version="1.0.0")

# Global TTS instance
tts = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"
    output_filename: str = None

@app.on_event("startup")
async def startup_event():
    global tts
    try:
        logger.info("Initializing Orpheus TTS model...")
        
        # Check for HF token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN environment variable not found!")
            logger.error("To fix this:")
            logger.error("   1. Get a token from: https://huggingface.co/settings/tokens")
            logger.error("   2. Request access to: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft")
            logger.error("   3. Set HUGGINGFACE_TOKEN in your .env file")
            raise ValueError("Missing HUGGINGFACE_TOKEN")
        
        # Allow model override via environment variable
        model_name = os.getenv('ORPHEUS_MODEL', 'canopylabs/orpheus-3b-0.1-ft')
        logger.info(f"Using model: {model_name}")
        
        tts = OrpheusTTS(model_name=model_name)
        logger.info("Orpheus TTS model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Orpheus TTS model: {e}")
        # Don't raise the exception - let the service start but return errors for requests
        tts = None

@app.get("/health")
def health_check():
    if tts is None:
        return {
            "status": "unhealthy", 
            "model": "orpheus",
            "error": "Model not loaded - check logs for authentication issues"
        }
    return {"status": "healthy", "model": "orpheus"}

@app.post("/synthesize")
def synthesize(request: TTSRequest):
    try:
        if tts is None:
            raise HTTPException(
                status_code=503, 
                detail="TTS model not loaded. Please check HUGGINGFACE_TOKEN and model access permissions."
            )
        
        output_filename = request.output_filename or f"orpheus_output_{hash(request.text) % 10000}.wav"
        
        # Support both Docker and local deployment
        # Docker: /tmp/orpheus_audio (mounted volume)
        # Local: outputs/orpheus (local directory)
        if os.path.exists("/tmp/orpheus_audio"):
            output_path = f"/tmp/orpheus_audio/{output_filename}"
        else:
            # Get the project root (2 levels up from services/modelOrpheus)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            output_dir = os.path.join(project_root, "outputs", "orpheus")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        
        output_file = tts.synthesize(request.text, voice=request.voice, output_path=output_path)
        
        return {
            "message": "Audio generated successfully",
            "file": output_file,
            "filename": output_filename,
            "host_path": f"outputs/orpheus/{output_filename}",
            "text": request.text,
            "voice": request.voice
        }
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        if "401" in str(e) or "gated" in str(e).lower():
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed. Please ensure HUGGINGFACE_TOKEN is valid and you have access to the model."
            )
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/synthesize")
def synthesize_get(text: str, voice: str = "tara"):
    """Backward compatibility endpoint"""
    request = TTSRequest(text=text, voice=voice)
    return synthesize(request)

@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """Serve generated audio files"""
    # Support both Docker and local deployment
    if os.path.exists("/tmp/orpheus_audio"):
        file_path = f"/tmp/orpheus_audio/{filename}"
    else:
        # Get the project root (2 levels up from services/modelOrpheus)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        file_path = os.path.join(project_root, "outputs", "orpheus", filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="audio/wav",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")