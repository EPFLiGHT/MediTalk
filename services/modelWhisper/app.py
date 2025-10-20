from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import logging
from pydantic import BaseModel
from typing import Optional
import subprocess
import json

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
            # Transcribe with Whisper using CLI to isolate from segfaults
            logger.info("Starting transcription process via whisper CLI...")
            
            # Get the base filename without extension
            base_name = os.path.basename(tmp_file_path).replace(".wav", "")
            output_dir = "/tmp"
            
            # Use whisper CLI in subprocess to avoid in-process segfaults
            cmd = [
                "whisper",
                tmp_file_path,
                "--model", os.getenv('WHISPER_MODEL', 'tiny'),
                "--language", "en",
                "--fp16", "False",
                "--output_format", "txt",
                "--output_dir", output_dir
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            logger.info(f"Return code: {result.returncode}")
            logger.info(f"Stdout: {result.stdout[:500]}")
            logger.info(f"Stderr: {result.stderr[:500]}")
            
            if result.returncode != 0:
                logger.error(f"Whisper CLI failed with code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                raise Exception(f"Whisper CLI failed: {result.stderr or result.stdout}")
            
            # Read the TXT output file
            txt_file = os.path.join(output_dir, f"{base_name}.txt")
            logger.info(f"Looking for output file: {txt_file}")
            
            if not os.path.exists(txt_file):
                # Try alternative naming
                txt_file = os.path.join(output_dir, f"{os.path.basename(tmp_file_path).replace('.wav', '')}.txt")
                logger.info(f"Trying alternative: {txt_file}")
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    text = f.read().strip()
                logger.info(f"Transcription completed: '{text[:100]}...'")
                # Clean up TXT file
                os.unlink(txt_file)
            else:
                # List files in output dir to debug
                files = os.listdir(output_dir)
                logger.error(f"Output file not found. Files in {output_dir}: {files[:10]}")
                text = result.stdout.strip() if result.stdout else "Transcription failed"
            
            return TranscriptionResponse(
                text=text,
                language="en",
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