from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from infer import OrpheusTTS
import os
import logging
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orpheus TTS Service", version="1.0.0")

# Global TTS instance
tts = None

# Task tracking for cancellation
active_tasks = {}  # task_id -> {"cancelled": bool}
tasks_lock = threading.Lock()

class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"
    output_filename: str = None
    task_id: Optional[str] = None

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

@app.get("/voices")
def get_voices():
    """Return list of available Orpheus voices"""
    return {
        "voices": [
            {"id": "tara", "name": "Tara", "gender": "female"},
            {"id": "leah", "name": "Leah", "gender": "female"},
            {"id": "jess", "name": "Jess", "gender": "female"},
            {"id": "mia", "name": "Mia", "gender": "female"},
            {"id": "zoe", "name": "Zoe", "gender": "female"},
            {"id": "leo", "name": "Leo", "gender": "male"},
            {"id": "dan", "name": "Dan", "gender": "male"},
            {"id": "zac", "name": "Zac", "gender": "male"}
        ],
        "default": "tara"
    }

def is_task_cancelled(task_id: str) -> bool:
    """Check if a task has been cancelled"""
    with tasks_lock:
        task = active_tasks.get(task_id)
        return task["cancelled"] if task else False

@app.post("/synthesize")
def synthesize(request: TTSRequest):
    task_id = request.task_id
    
    # Register task if task_id provided
    if task_id:
        with tasks_lock:
            active_tasks[task_id] = {"cancelled": False}
        logger.info(f"Orpheus TTS task {task_id} started")
    
    try:
        if tts is None:
            raise HTTPException(
                status_code=503, 
                detail="TTS model not loaded. Please check HUGGINGFACE_TOKEN and model access permissions."
            )
        
        # Check cancellation before starting
        if task_id and is_task_cancelled(task_id):
            logger.info(f"Orpheus TTS task {task_id} cancelled before synthesis")
            raise HTTPException(status_code=499, detail="Task cancelled by user")
        
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
        
        output_file = tts.synthesize(request.text, voice=request.voice, output_path=output_path, task_id=task_id)
        
        # Check cancellation after synthesis
        if task_id and is_task_cancelled(task_id):
            logger.info(f"Orpheus TTS task {task_id} cancelled after synthesis")
            raise HTTPException(status_code=499, detail="Task cancelled by user")
        
        return {
            "message": "Audio generated successfully",
            "file": output_file,
            "filename": output_filename,
            "host_path": f"outputs/orpheus/{output_filename}",
            "text": request.text,
            "voice": request.voice
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        if "401" in str(e) or "gated" in str(e).lower():
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed. Please ensure HUGGINGFACE_TOKEN is valid and you have access to the model."
            )
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    finally:
        # Clean up task
        if task_id:
            with tasks_lock:
                active_tasks.pop(task_id, None)
            logger.info(f"Orpheus TTS task {task_id} cleaned up")

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

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel an active task"""
    with tasks_lock:
        task = active_tasks.get(task_id)
        if task:
            task["cancelled"] = True
            logger.info(f"Orpheus TTS task {task_id} marked for cancellation")
            return {"status": "cancelled", "task_id": task_id}
        else:
            logger.warning(f"Orpheus TTS task {task_id} not found (may have already completed)")
            return {"status": "not_found", "task_id": task_id}