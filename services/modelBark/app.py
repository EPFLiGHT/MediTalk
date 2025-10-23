from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import logging
import numpy as np
from scipy.io.wavfile import write as write_wav
from typing import Optional
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bark TTS Service", version="1.0.0")

# Global BARK model instances
bark_generate_audio = None
bark_preload_models = None

# Task tracking for cancellation
active_tasks = {}  # task_id -> {"cancelled": bool}
tasks_lock = threading.Lock()

class TTSRequest(BaseModel):
    text: str
    voice: str = "v2/en_speaker_6"  # BARK voice preset
    output_filename: Optional[str] = None
    task_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global bark_generate_audio, bark_preload_models
    try:
        logger.info("Initializing Bark TTS model...")
        
        # Workaround for PyTorch 2.6+ weights_only issue with Bark models
        # Bark is from Suno AI (trusted source), so we can safely use weights_only=False
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        # Import BARK modules
        from bark import SAMPLE_RATE, generate_audio, preload_models
        
        bark_generate_audio = generate_audio
        bark_preload_models = preload_models
        
        # Preload models (downloads on first run)
        logger.info("Preloading BARK models (this may take a while on first run)...")
        bark_preload_models()
        
        # Restore original torch.load
        torch.load = original_load
        
        logger.info("Bark TTS model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Bark TTS model: {e}")
        import traceback
        traceback.print_exc()
        bark_generate_audio = None
        bark_preload_models = None

@app.get("/health")
def health_check():
    if bark_generate_audio is None:
        return {"status": "unhealthy", "model": "bark-tts", "error": "Model not loaded"}
    return {"status": "healthy", "model": "bark-tts"}

def split_text_for_bark(text: str, max_chars: int = 250) -> list:
    """
    Split text into chunks suitable for Bark TTS generation.
    Bark works best with chunks under ~250 characters.
    Splits on sentence boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chars
        if len(current_chunk) + len(sentence) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
            else:
                # Single sentence is too long, split it by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chars:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def is_task_cancelled(task_id: str) -> bool:
    """Check if a task has been cancelled"""
    with tasks_lock:
        task = active_tasks.get(task_id)
        return task["cancelled"] if task else False

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text using Bark TTS.
    Automatically splits long texts into chunks to avoid truncation.
    """
    task_id = request.task_id
    
    # Register task if task_id provided
    if task_id:
        with tasks_lock:
            active_tasks[task_id] = {"cancelled": False}
        logger.info(f"Bark TTS task {task_id} started")
    
    try:
        if bark_generate_audio is None:
            raise HTTPException(status_code=503, detail="Bark TTS model not loaded")
        
        # Check cancellation before starting
        if task_id and is_task_cancelled(task_id):
            logger.info(f"Bark TTS task {task_id} cancelled before synthesis")
            raise HTTPException(status_code=499, detail="Task cancelled by user")
        
        logger.info(f"Synthesizing speech for text: {request.text[:50]}... (length: {len(request.text)} chars)")
        logger.info(f"Using voice preset: {request.voice}")
        
        # Import BARK constants
        from bark import SAMPLE_RATE
        
        # Split text into manageable chunks for Bark
        text_chunks = split_text_for_bark(request.text, max_chars=250)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        # Generate audio for each chunk
        audio_arrays = []
        for i, chunk in enumerate(text_chunks):
            # Check cancellation before each chunk
            if task_id and is_task_cancelled(task_id):
                logger.info(f"Bark TTS task {task_id} cancelled during chunk {i+1}/{len(text_chunks)}")
                raise HTTPException(status_code=499, detail="Task cancelled by user")
            
            logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            chunk_audio = bark_generate_audio(
                chunk,
                history_prompt=request.voice  # Voice preset (e.g., "v2/en_speaker_6")
            )
            audio_arrays.append(chunk_audio)
        
        # Check cancellation after generation
        if task_id and is_task_cancelled(task_id):
            logger.info(f"Bark TTS task {task_id} cancelled after audio generation")
            raise HTTPException(status_code=499, detail="Task cancelled by user")
        
        # Concatenate all audio chunks
        if len(audio_arrays) > 1:
            audio_array = np.concatenate(audio_arrays)
            logger.info(f"Concatenated {len(audio_arrays)} audio chunks")
        else:
            audio_array = audio_arrays[0]
        
        # Create output directory if it doesn't exist
        output_dir = "../../outputs/bark"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if request.output_filename:
            filename = request.output_filename
        else:
            import random
            filename = f"bark_output_{random.randint(1000, 9999)}.wav"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save audio to file
        write_wav(filepath, SAMPLE_RATE, audio_array)
        
        logger.info(f"Audio saved to: {filepath}")
        
        return {
            "status": "success",
            "filename": filename,
            "filepath": filepath,
            "url": f"/audio/{filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    finally:
        # Clean up task
        if task_id:
            with tasks_lock:
                active_tasks.pop(task_id, None)
            logger.info(f"Bark TTS task {task_id} cleaned up")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve generated audio files
    """
    filepath = os.path.join("../../outputs/bark", filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename
    )

@app.get("/voices")
def list_voices():
    """
    List available BARK voice presets
    """
    # BARK voice presets (multilingual)
    voices = {
        "english": [
            "v2/en_speaker_0",
            "v2/en_speaker_1",
            "v2/en_speaker_2",
            "v2/en_speaker_3",
            "v2/en_speaker_4",
            "v2/en_speaker_5",
            "v2/en_speaker_6",
            "v2/en_speaker_7",
            "v2/en_speaker_8",
            "v2/en_speaker_9"
        ],
        "multilingual": [
            "v2/de_speaker_0",  # German
            "v2/de_speaker_1",
            "v2/es_speaker_0",  # Spanish
            "v2/fr_speaker_0",  # French
            "v2/hi_speaker_0",  # Hindi
            "v2/it_speaker_0",  # Italian
            "v2/ja_speaker_0",  # Japanese
            "v2/ko_speaker_0",  # Korean
            "v2/pl_speaker_0",  # Polish
            "v2/pt_speaker_0",  # Portuguese
            "v2/ru_speaker_0",  # Russian
            "v2/tr_speaker_0",  # Turkish
            "v2/zh_speaker_0"   # Chinese
        ]
    }
    
    return {
        "status": "success",
        "voices": voices,
        "default": "v2/en_speaker_6"
    }

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel an active task"""
    with tasks_lock:
        task = active_tasks.get(task_id)
        if task:
            task["cancelled"] = True
            logger.info(f"Bark TTS task {task_id} marked for cancellation")
            return {"status": "cancelled", "task_id": task_id}
        else:
            logger.warning(f"Bark TTS task {task_id} not found (may have already completed)")
            return {"status": "not_found", "task_id": task_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
