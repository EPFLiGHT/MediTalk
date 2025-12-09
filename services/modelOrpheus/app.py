from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from infer import OrpheusTTS
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orpheus TTS Service", version="1.0.0")

# Global TTS instances (one for each language)
tts_instances = {
    "en": None,  # English model
    "fr": None   # French model
}

# Model configurations
MODELS = {
    "en": "canopylabs/orpheus-3b-0.1-ft",
    "fr": "canopylabs/3b-fr-ft-research_release"
}

class TTSRequest(BaseModel):
    conversation_path: str  # Path to conversation JSON file
    voice: str = "tara"
    output_filename: str = None
    generate_in_parallel: bool = True  # New parameter for parallel generation
    language: str = "en"  # Language: "en" for English, "fr" for French
    text: Optional[str] = None  # Optional direct text (for backward compatibility)

@app.on_event("startup")
async def startup_event():
    global tts_instances
    try:
        logger.info("Initializing Orpheus TTS models...")
        
        # Check for HF token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN environment variable not found!")
            logger.error("To fix this:")
            logger.error("   1. Get a token from: https://huggingface.co/settings/tokens")
            logger.error("   2. Request access to: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft")
            logger.error("   3. Request access to: https://huggingface.co/canopylabs/3b-fr-ft-research_release")
            logger.error("   4. Set HUGGINGFACE_TOKEN in your .env file")
            raise ValueError("Missing HUGGINGFACE_TOKEN")
        
        # Parallel configuration from environment variables
        max_parallel = int(os.getenv('ORPHEUS_MAX_PARALLEL_CHUNKS', '16'))
        logger.info(f"Parallel configuration: max_parallel_chunks={max_parallel}")
        
        # Determine which languages to load based on environment variable
        # Default: load only English model
        languages_to_load = os.getenv('ORPHEUS_LANGUAGES', 'en').split(',')
        
        for lang in languages_to_load:
            lang = lang.strip()
            if lang in MODELS:
                try:
                    model_name = MODELS[lang]
                    logger.info(f"Loading {lang.upper()} model: {model_name}")
                    
                    tts_instances[lang] = OrpheusTTS(
                        model_name=model_name,
                        max_parallel_chunks=max_parallel
                    )
                    logger.info(f"✓ {lang.upper()} model loaded successfully!")
                    
                except Exception as e:
                    logger.error(f"Failed to load {lang.upper()} model: {e}")
                    tts_instances[lang] = None
            else:
                logger.warning(f"Unknown language '{lang}' - skipping")
        
        # Log available languages
        available = [lang for lang, tts in tts_instances.items() if tts is not None]
        if available:
            logger.info(f"Available languages: {', '.join(available)}")
        else:
            logger.error("No TTS models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Orpheus TTS models: {e}")
        # Don't raise the exception - let the service start but return errors for requests

@app.get("/health")
def health_check():
    available_languages = [lang for lang, tts in tts_instances.items() if tts is not None]
    
    if not available_languages:
        return {
            "status": "unhealthy", 
            "model": "orpheus",
            "error": "No models loaded - check logs for authentication issues",
            "available_languages": []
        }
    
    return {
        "status": "healthy", 
        "model": "orpheus",
        "available_languages": available_languages
    }

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
        # Extract text from conversation JSON or use direct text
        text_to_synthesize = None
        
        if request.conversation_path:
            # Read conversation JSON and extract last assistant message
            try:
                with open(request.conversation_path, 'r') as f:
                    conversation = json.load(f)
                
                # Find last assistant message
                messages = conversation.get('messages', [])
                for message in reversed(messages):
                    if message.get('role') == 'assistant':
                        # Extract text from content array
                        for content_item in message.get('content', []):
                            if content_item.get('type') == 'text':
                                text_to_synthesize = content_item.get('data')
                                break
                        if text_to_synthesize:
                            break
                
                if not text_to_synthesize:
                    raise HTTPException(
                        status_code=400,
                        detail="No assistant text found in conversation"
                    )
                    
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation file not found: {request.conversation_path}"
                )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in conversation file: {str(e)}"
                )
        elif request.text:
            # Use direct text for backward compatibility
            text_to_synthesize = request.text
        else:
            raise HTTPException(
                status_code=400,
                detail="Either conversation_path or text must be provided"
            )
        
        # Get the TTS instance for the requested language
        language = request.language.lower()
        tts = tts_instances.get(language)
        
        if tts is None:
            available = [lang for lang, instance in tts_instances.items() if instance is not None]
            raise HTTPException(
                status_code=400,
                detail=f"Language '{language}' not available. Available languages: {', '.join(available)}"
            )
        
        output_filename = request.output_filename or f"orpheus_output_{language}_{hash(text_to_synthesize) % 10000}.wav"
        
        # Support both Docker and local deployment
        if os.path.exists("/tmp/orpheus_audio"):
            output_path = f"/tmp/orpheus_audio/{output_filename}"
        else:
            output_dir = os.path.join("../../outputs", "orpheus")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        
        # Pass the generate_in_parallel parameter to the TTS engine
        output_file = tts.synthesize(
            text_to_synthesize, 
            voice=request.voice, 
            output_path=output_path,
            generate_in_parallel=request.generate_in_parallel
        )
        
        return JSONResponse(content={
            "status": "success",
            "filename": output_filename,
            "service": "orpheus",
            "message": "Audio generated successfully",
            "parallel_mode": request.generate_in_parallel,
            "language": language
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Orpheus TTS synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)