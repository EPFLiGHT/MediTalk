from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
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
        if os.path.exists("/tmp/orpheus_audio"):
            output_path = f"/tmp/orpheus_audio/{output_filename}"
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            output_dir = os.path.join(project_root, "outputs", "orpheus")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        
        output_file = tts.synthesize(request.text, voice=request.voice, output_path=output_path)
        
        return JSONResponse(content={
            "status": "success",
            "audio_file": output_path,
            "message": "Audio generated successfully"
        })
    except Exception as e:
        logger.error(f"Error in Orpheus TTS synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
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
        # DON'T clean up task immediately - keep it for a few seconds to allow in-flight cancel requests
        # The task will be cleaned up by a background task or timeout
        if task_id:
            # Schedule cleanup after a delay to allow cancel requests to arrive
            async def delayed_cleanup():
                await asyncio.sleep(5)  # Wait 5 seconds before cleaning up
                with tasks_lock:
                    active_tasks.pop(task_id, None)
                logger.info(f"Orpheus TTS task {task_id} cleaned up (delayed)")
            
            # Create task for delayed cleanup (fire and forget)
            asyncio.create_task(delayed_cleanup())
            logger.info(f"Orpheus TTS task {task_id} scheduled for cleanup in 5 seconds")

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
    logger.info(f"CANCEL REQUEST RECEIVED for task {task_id}")
    with tasks_lock:
        logger.info(f"Active tasks BEFORE cancel: {active_tasks}")
        task = active_tasks.get(task_id)
        if task:
            logger.info(f"Task {task_id} found, BEFORE setting cancelled: {task}")
            task["cancelled"] = True
            # Verify it was actually set
            logger.info(f"Task {task_id} AFTER setting cancelled: {task}")
            logger.info(f"Active tasks AFTER cancel: {active_tasks}")
            logger.info(f"Orpheus TTS task {task_id} MARKED FOR CANCELLATION - cancelled={task['cancelled']}")
            return {"status": "cancelled", "task_id": task_id}
        else:
            logger.warning(f"Orpheus TTS task {task_id} NOT FOUND in active tasks")
            logger.warning(f"Available task IDs: {list(active_tasks.keys())}")
            return {"status": "not_found", "task_id": task_id}