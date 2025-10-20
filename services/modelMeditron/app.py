from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from meditron import MeditronLLM
import requests
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meditron Medical AI Service", version="1.0.0")

# Service URLs - support both Docker and local deployment
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")

# Global model instance
meditron = None

class QuestionRequest(BaseModel):
    question: str
    max_length: int = 512
    temperature: float = 0.7
    generate_audio: bool = True
    voice: str = "tara"

from typing import Optional

class MedicalResponse(BaseModel):
    question: str
    answer: str
    audio_file: Optional[str] = None
    audio_url: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global meditron
    try:
        logger.info("Initializing Meditron medical AI...")
        meditron = MeditronLLM()
        logger.info("Meditron medical AI loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Meditron: {e}")
        meditron = None

@app.get("/health")
def health_check():
    if meditron is None:
        return {
            "status": "unhealthy", 
            "model": "meditron-7b",
            "error": "Model not loaded"
        }
    return {"status": "healthy", "model": "meditron-7b"}

@app.post("/ask", response_model=MedicalResponse)
def ask_medical_question(request: QuestionRequest):
    """Ask a medical question and optionally get audio response"""
    try:
        if meditron is None:
            raise HTTPException(status_code=503, detail="Meditron model not loaded")
        
        # Generate text response using Meditron
        logger.info(f"Processing question: {request.question[:50]}...")
        answer = meditron.generate_response(
            request.question,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        audio_file = None
        audio_url = None
        
        # Generate audio if requested
        if request.generate_audio:
            try:
                logger.info("Generating audio response using Orpheus...")
                orpheus_response = requests.post(
                    f"{ORPHEUS_URL}/synthesize",
                    json={
                        "text": answer,
                        "voice": request.voice
                    },
                    timeout=600  # Increased to 10 minutes for complex audio generation
                )
                
                if orpheus_response.status_code == 200:
                    orpheus_data = orpheus_response.json()
                    audio_file = orpheus_data.get("filename")
                    audio_url = f"http://localhost:8080/audio/{audio_file}" if audio_file else None
                    logger.info(f"Audio generated: {audio_file}")
                else:
                    logger.warning(f"Audio generation failed: {orpheus_response.status_code}")
                    
            except Exception as audio_error:
                logger.warning(f"Audio generation failed: {audio_error}")
                # Continue without audio - text response is still valuable
        
        return MedicalResponse(
            question=request.question,
            answer=answer,
            audio_file=audio_file,
            audio_url=audio_url
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/ask")
def ask_simple(question: str, generate_audio: bool = True):
    """Simple GET endpoint for quick testing"""
    request = QuestionRequest(question=question, generate_audio=generate_audio)
    return ask_medical_question(request)