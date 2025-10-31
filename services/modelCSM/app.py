from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
import torch
import torchaudio
import tempfile
import hashlib
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CSM TTS Service", version="1.0.0")

# Global CSM generator instance
csm_generator = None

class ConversationSegment(BaseModel):
    """A segment of conversation history for CSM context"""
    text: str
    speaker: int  # 0 or 1 for two speakers
    audio_url: Optional[str] = None  # URL to audio file if available

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0  # Speaker ID (0 or 1)
    conversation_history: Optional[List[ConversationSegment]] = None
    max_audio_length_ms: int = 10000  # Per chunk
    output_filename: Optional[str] = None

def split_text_for_csm(text: str, max_chars: int = 200) -> list:
    """
    Split text into chunks suitable for CSM TTS generation.
    Splits on sentence boundaries when possible.
    Similar to Bark chunking but more conservative to leave room for context.
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

@app.on_event("startup")
async def startup_event():
    global csm_generator
    try:
        logger.info("Initializing CSM (Conversational Speech Model)...")
        
        # Check for HF token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN environment variable not found!")
            logger.error("To fix this:")
            logger.error("   1. Get a token from: https://huggingface.co/settings/tokens")
            logger.error("   2. Request access to: https://huggingface.co/sesame/csm-1b")
            logger.error("   3. Request access to: https://huggingface.co/meta-llama/Llama-3.2-1B")
            logger.error("   4. Set HUGGINGFACE_TOKEN in your .env file")
            raise ValueError("Missing HUGGINGFACE_TOKEN")
        
        # Disable lazy compilation in Mimi as per CSM docs
        os.environ['NO_TORCH_COMPILE'] = '1'
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load CSM generator
        from generator import load_csm_1b
        csm_generator = load_csm_1b(device=device)
        
        logger.info("CSM model loaded successfully!")
        logger.info(f"Sample rate: {csm_generator.sample_rate} Hz")
        
    except Exception as e:
        logger.error(f"Failed to load CSM model: {e}")
        import traceback
        traceback.print_exc()
        csm_generator = None

@app.get("/health")
def health_check():
    if csm_generator is None:
        return {
            "status": "unhealthy", 
            "model": "csm-1b",
            "error": "Model not loaded - check logs for authentication issues"
        }
    return {
        "status": "healthy", 
        "model": "csm-1b",
        "sample_rate": csm_generator.sample_rate if csm_generator else None
    }

def load_audio_from_path(audio_path: str, target_sample_rate: int, max_duration_sec: float = 3.0):
    """
    Load audio file and resample to target sample rate.
    Truncates audio to max_duration_sec to prevent context overflow.
    """
    try:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        # Resample if needed
        if sample_rate != target_sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
        else:
            audio_tensor = audio_tensor.squeeze(0)
        
        # Truncate to max duration to save context space
        max_samples = int(max_duration_sec * target_sample_rate)
        if audio_tensor.shape[0] > max_samples:
            # Take the last max_duration_sec seconds (more recent audio)
            audio_tensor = audio_tensor[-max_samples:]
            logger.info(f"  Truncated audio to {max_duration_sec}s for context efficiency")
        
        return audio_tensor
    except Exception as e:
        logger.error(f"Error loading audio from {audio_path}: {e}")
        return None

def download_audio_url(url: str, target_sample_rate: int, max_duration_sec: float = 3.0):
    """
    Download audio from URL and load it.
    Truncates to max_duration_sec to prevent context overflow.
    """
    import requests
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Load audio (with truncation)
        audio = load_audio_from_path(tmp_path, target_sample_rate, max_duration_sec)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return audio
    except Exception as e:
        logger.error(f"Error downloading audio from {url}: {e}")
        return None

@app.post("/synthesize")
def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text using CSM with conversation context.
    CSM generates more natural, contextually appropriate speech when provided
    with conversation history including both text and audio.
    """
    try:
        if csm_generator is None:
            raise HTTPException(
                status_code=503, 
                detail="CSM model not loaded. Please check HUGGINGFACE_TOKEN and model access permissions."
            )
        
        logger.info(f"Synthesizing speech for: {request.text[:100]}...")
        logger.info(f"Speaker: {request.speaker}, Context segments: {len(request.conversation_history) if request.conversation_history else 0}")
        
        # Import Segment class for context
        from generator import Segment
        
        # Build context from conversation history
        logger.info("Building conversation context...")
        context_segments = []
        if request.conversation_history:
            # Limit to last 3 conversation turns to prevent context overflow
            # Each audio segment can be ~300-400 tokens (3 sec * 24000 Hz / 1920 samples per token)
            recent_history = request.conversation_history[-3:]
            logger.info(f"Using last {len(recent_history)} of {len(request.conversation_history)} conversation turns")
            
            for seg in recent_history:
                audio_tensor = None
                
                # Try to load audio if URL is provided
                if seg.audio_url:
                    # Check if it's a local file path or URL
                    if seg.audio_url.startswith('http://') or seg.audio_url.startswith('https://'):
                        audio_tensor = download_audio_url(seg.audio_url, csm_generator.sample_rate, max_duration_sec=3.0)
                    else:
                        # Local file path - handle both relative and absolute
                        if not os.path.isabs(seg.audio_url):
                            # Try outputs directory
                            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                            audio_path = os.path.join(project_root, "outputs", seg.audio_url)
                        else:
                            audio_path = seg.audio_url
                        
                        if os.path.exists(audio_path):
                            audio_tensor = load_audio_from_path(audio_path, csm_generator.sample_rate, max_duration_sec=3.0)
                
                # Only add segments with valid audio to context
                # CSM requires audio for all context segments (text-only not supported in Segment)
                if audio_tensor is not None:
                    segment = Segment(
                        text=seg.text,
                        speaker=seg.speaker,
                        audio=audio_tensor
                    )
                    context_segments.append(segment)
                    logger.info(f"  Added context with audio: speaker={seg.speaker}, text='{seg.text[:50]}...'")
                else:
                    logger.info(f"  Skipped context (no audio): speaker={seg.speaker}, text='{seg.text[:50]}...'")
        
        # Split text into chunks if needed
        text_chunks = split_text_for_csm(request.text, max_chars=200)
        logger.info(f"Split text into {len(text_chunks)} chunks for generation")
        
        # Generate audio for each chunk
        audio_arrays = []
        import time
        start_time = time.time()
        
        # Keep only the original conversation context, don't accumulate chunks
        original_context_count = len(context_segments)
        context_skipped = False  # Track if we had to skip context due to overflow
        
        # Try with context first, if it fails due to context overflow, retry without context
        try:
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: {chunk[:80]}...")
                
                chunk_audio = csm_generator.generate(
                    text=chunk,
                    speaker=request.speaker,
                    context=context_segments,
                    max_audio_length_ms=request.max_audio_length_ms,
                )
                
                audio_arrays.append(chunk_audio.cpu().numpy())
                
                # After first chunk, add ONLY the most recent chunk to context for continuity
                # Remove the previous generated chunk to prevent context overflow
                if i < len(text_chunks) - 1:
                    from generator import Segment
                    
                    # Remove any previously added generated chunks (keep only original context)
                    context_segments = context_segments[:original_context_count]
                    
                    # Add only the current chunk to context
                    context_segments.append(Segment(
                        text=chunk,
                        speaker=request.speaker,
                        audio=chunk_audio
                    ))
                    logger.info(f"  Updated context with current chunk (context size: {len(context_segments)})")
        
        except ValueError as e:
            # Check if it's a context overflow error
            if "Inputs too long" in str(e):
                logger.warning(f"/!\ Context overflow error: {e}")
                logger.warning("Retrying audio generation WITHOUT conversation context...")
                context_skipped = True
                
                # Retry without any context
                audio_arrays = []
                for i, chunk in enumerate(text_chunks):
                    logger.info(f"Generating chunk {i+1}/{len(text_chunks)} (no context): {chunk[:80]}...")
                    
                    chunk_audio = csm_generator.generate(
                        text=chunk,
                        speaker=request.speaker,
                        context=[],  # Empty context
                        max_audio_length_ms=request.max_audio_length_ms,
                    )
                    
                    audio_arrays.append(chunk_audio.cpu().numpy())
                
                logger.info("✓ Successfully generated audio without context")
            else:
                # Re-raise if it's a different ValueError
                raise
        
        # Concatenate all audio chunks
        if len(audio_arrays) > 1:
            audio = torch.from_numpy(np.concatenate(audio_arrays))
            logger.info(f"✓ Concatenated {len(audio_arrays)} audio chunks")
        else:
            audio = audio_arrays[0]
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
        
        generation_time = time.time() - start_time
        logger.info(f"✓ Complete audio generated in {generation_time:.2f}s ({len(audio)/csm_generator.sample_rate:.2f}s of audio)")

        
        # Create output directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        output_dir = os.path.join(project_root, "outputs", "csm")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if request.output_filename:
            output_filename = request.output_filename
        else:
            # Create unique filename based on timestamp and text hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_hash = hashlib.md5(request.text.encode()).hexdigest()[:8]
            output_filename = f"csm_{timestamp}_{text_hash}.wav"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save audio file
        torchaudio.save(
            output_path, 
            audio.unsqueeze(0).cpu(), 
            csm_generator.sample_rate
        )
        
        logger.info(f"Audio saved to: {output_path}")
        
        message = "Audio generated successfully with conversation context"
        if context_skipped:
            message = "Audio generated without context (conversation context was too long)"
        
        return JSONResponse(content={
            "status": "success",
            "filename": output_filename,
            "audio_file": output_path,
            "sample_rate": csm_generator.sample_rate,
            "message": message,
            "context_skipped": context_skipped
        })
        
    except Exception as e:
        logger.error(f"Error in CSM synthesis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/synthesize")
def synthesize_get(text: str, speaker: int = 0):
    """Backward compatibility endpoint for simple synthesis without context"""
    request = TTSRequest(text=text, speaker=speaker)
    return synthesize_speech(request)

@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """Serve generated audio files"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    file_path = os.path.join(project_root, "outputs", "csm", filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="audio/wav",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/upload_context_audio")
async def upload_context_audio(audio_file: UploadFile = File(...)):
    """
    Upload an audio file to be used as context in future synthesis requests.
    Returns the local path that can be used in ConversationSegment.audio_url
    """
    try:
        # Create temp directory for context audio
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        context_dir = os.path.join(project_root, "outputs", "csm", "context")
        os.makedirs(context_dir, exist_ok=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"context_{timestamp}_{audio_file.filename}"
        file_path = os.path.join(context_dir, filename)
        
        with open(file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        logger.info(f"Context audio uploaded: {file_path}")
        
        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "path": file_path,
            "relative_path": f"csm/context/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error uploading context audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
