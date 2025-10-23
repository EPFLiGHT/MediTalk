from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from meditron import MeditronLLM
import requests
import logging
import os
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meditron Medical AI Service", version="1.0.0")

# Service URLs - support both Docker and local deployment
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
BARK_URL = os.getenv("BARK_URL", "http://localhost:5008")

# Global model instance
meditron = None

def clean_text_for_tts(text: str) -> str:
    """
    Remove markdown formatting from text before sending to TTS.
    This prevents TTS from reading "asterisk" or other markup symbols.
    """
    # Remove bold/italic markers: **text**, __text__, *text*, _text_
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'__([^_]+)__', r'\1', text)        # __bold__
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)       # *italic*
    text = re.sub(r'_([^_]+)_', r'\1', text)          # _italic_
    
    # Remove markdown headers: # Header, ## Header, etc.
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown list markers: - item, * item, 1. item
    text = re.sub(r'^\s*[-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove inline code markers: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    return text.strip()


class QuestionRequest(BaseModel):
    question: str
    max_length: int = 512
    temperature: float = 0.7
    generate_audio: bool = True
    voice: str = "tara"
    tts_service: str = "orpheus"  # "orpheus" or "bark"

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
        
        # Save text response to file
        try:
            output_dir = "../../outputs/text"
            os.makedirs(output_dir, exist_ok=True)
            
            import random
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            text_filename = f"meditron_response_{timestamp}_{random.randint(1000, 9999)}.txt"
            text_filepath = os.path.join(output_dir, text_filename)
            
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(f"Question:\n{request.question}\n\n")
                f.write(f"Answer:\n{answer}\n")
            
            logger.info(f"Text response saved to: {text_filepath}")
        except Exception as save_error:
            logger.warning(f"Failed to save text response: {save_error}")
        
        audio_file = None
        audio_url = None
        
        # Generate audio if requested
        if request.generate_audio:
            try:
                # Select TTS service based on request parameter
                tts_url = BARK_URL if request.tts_service == "bark" else ORPHEUS_URL
                tts_name = request.tts_service.capitalize()
                
                # Clean markdown formatting from text before TTS
                clean_answer = clean_text_for_tts(answer)
                logger.info(f"Generating audio response using {tts_name}...")
                logger.info(f"Cleaned {len(answer) - len(clean_answer)} characters of markdown formatting for TTS")
                
                tts_response = requests.post(
                    f"{tts_url}/synthesize",
                    json={
                        "text": clean_answer,
                        "voice": request.voice
                    },
                    timeout=600  # Increased to 10 minutes for complex audio generation
                )
                
                if tts_response.status_code == 200:
                    tts_data = tts_response.json()
                    audio_file = tts_data.get("filename")
                    audio_url = f"http://localhost:8080/audio/{audio_file}" if audio_file else None
                    logger.info(f"Audio generated by {tts_name}: {audio_file}")
                else:
                    logger.warning(f"Audio generation failed: {tts_response.status_code}")
                    
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