from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict
import requests
import logging
import os
import torch
import re
from transformers import AutoTokenizer
from PIL import Image
import io
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MultiMeditron Multimodal Medical AI Service", version="1.0.0")

# Service URLs
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
BARK_URL = os.getenv("BARK_URL", "http://localhost:5008")

# Global model instance (following MultiMeditron README)
model = None
tokenizer = None
collator = None
image_loader = None
ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
attachment_token_idx = None

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

class MedicalResponse(BaseModel):
    question: str
    answer: str
    audio_file: Optional[str] = None
    audio_url: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, attachment_token_idx, collator, image_loader
    try:
        logger.info("Initializing MultiMeditron medical AI...")
        
        # Get model path from environment variable
        # Default to the private HuggingFace model
        model_path = os.getenv("MULTIMEDITRON_MODEL", "ClosedMeditron/Mulimeditron-End2End-CLIP-medical")
        
        # Get HuggingFace token for accessing private model
        hf_token = os.getenv("MULTIMEDITRON_HF_TOKEN", None)
        if not hf_token:
            logger.warning("MULTIMEDITRON_HF_TOKEN not set. Cannot access private model.")
            logger.warning("Please set MULTIMEDITRON_HF_TOKEN in .env file")
            return
        
        # Following the official documentation: https://epflight.github.io/MultiMeditron/guides/quickstart.html
        logger.info(f"Loading tokenizer from base LLM...")
        
        # Load tokenizer (from base Llama model, as shown in docs)
        base_llm = os.getenv("BASE_LLM", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(
            base_llm, 
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": [ATTACHMENT_TOKEN]}
        tokenizer.add_special_tokens(special_tokens)
        attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)
        logger.info(f"Attachment token '{ATTACHMENT_TOKEN}' index: {attachment_token_idx}")
        
        # Load MultiMeditron model (following official docs)
        logger.info(f"Loading MultiMeditron model from {model_path}...")
        from multimeditron.model.model import MultiModalModelForCausalLM
        from multimeditron.dataset.loader import FileSystemImageLoader
        from multimeditron.model.data_loader import DataCollatorForMultimodal
        
        # Use device_map="auto" as shown in official docs
        model = MultiModalModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            token=hf_token
        )
        model.eval()  # Set to evaluation mode (from official docs)
        logger.info(f"Model loaded with device_map='auto'")
        
        # Setup image loader (from official docs - note: FileSystemImageLoader not FileSystemImageRegistry)
        image_loader = FileSystemImageLoader(base_path=os.getcwd())
        
        # Setup collator (following official docs)
        collator = DataCollatorForMultimodal(
            tokenizer=tokenizer,
            tokenizer_type="llama",
            modality_processors=model.processors(),
            modality_loaders={"image": image_loader},
            attachment_token_idx=attachment_token_idx,
            add_generation_prompt=True
        )
        
        logger.info("✅ MultiMeditron fully loaded and ready!")
        
    except ImportError as e:
        logger.error(f"Failed to import MultiMeditron modules: {e}")
        logger.error("Make sure you have run: pip install -e ./multimeditron")
        logger.error("Service will continue in fallback mode.")
    except Exception as e:
        logger.error(f"Failed to initialize MultiMeditron: {e}")
        logger.error("Service will continue in fallback mode.")
        import traceback
        traceback.print_exc()

@app.get("/health")
def health_check():
    if model is not None and tokenizer is not None and collator is not None:
        status = "ready"
    elif tokenizer is not None:
        status = "partial"
    else:
        status = "not_ready"
    
    return {
        "status": status,
        "service": "MultiMeditron Medical AI",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "collator_ready": collator is not None,
        "attachment_token": ATTACHMENT_TOKEN
    }

@app.post("/ask", response_model=MedicalResponse)
async def ask_question(request: QuestionRequest):
    """
    Text-only medical question endpoint (compatible with existing Meditron API)
    """
    try:
        if not model or not tokenizer or not collator:
            # Fallback mode
            logger.warning("Model not fully loaded, using fallback response")
            answer = generate_fallback_response(request.question)
            return MedicalResponse(question=request.question, answer=answer)
        
        logger.info("Generating text response...")
        
        # Generate answer using MultiMeditron (text-only, no modalities)
        answer = generate_multimeditron_response(
            question=request.question,
            modalities=[],  # No images for text-only
            temperature=request.temperature,
            max_length=request.max_length
        )
        
        response_data = {
            "question": request.question,
            "answer": answer,
        }
        
        # Generate audio if requested
        if request.generate_audio:
            try:
                logger.info(f"Generating audio with {request.tts_service}...")
                audio_result = generate_audio(answer, request.voice, request.tts_service)
                if audio_result:
                    response_data["audio_file"] = audio_result.get("filename")
                    response_data["audio_url"] = audio_result.get("url")
            except Exception as e:
                logger.warning(f"Failed to generate audio: {e}")
        
        logger.info("Request completed successfully")
        return MedicalResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_multimeditron_response(
    question: str, 
    modalities: List[Dict], 
    temperature: float = 0.7,
    max_length: int = 512
) -> str:
    """
    Generate response using MultiMeditron model
    Following the exact pattern from MultiMeditron README
    """
    global model, tokenizer, collator, attachment_token_idx
    
    # Build the sample following README format
    conversations = [{
        "role": "user",
        "content": question
    }]
    
    sample = {
        "conversations": conversations,
        "modalities": modalities
    }
    
    logger.info(f"Generating response for question: {question[:100]}")
    logger.info(f"With {len(modalities)} modalities")
    
    # Create batch using collator (following README)
    batch = collator([sample])
    
    # Generate using model (following README)
    with torch.no_grad():
        outputs = model.generate(batch=batch, temperature=temperature)
    
    # Decode output (following README)
    response_text = tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    
    logger.info(f"Generated response: {response_text[:100]}...")
    
    return response_text

def generate_fallback_response(question: str) -> str:
    """
    Fallback response when model is not loaded yet
    """
    return (
        f"[MultiMeditron Fallback Mode] "
        f"This is a placeholder response. The MultiMeditron model is being integrated. "
        f"Your question was: '{question}'. "
        f"Once the full model is loaded, you'll get real medical AI responses with multimodal support."
    )

def generate_audio(text: str, voice: str, tts_service: str = "orpheus") -> Optional[Dict]:
    """
    Generate audio using specified TTS service (Orpheus or Bark)
    """
    try:
        # Clean markdown formatting from text before TTS
        clean_text = clean_text_for_tts(text)
        logger.info(f"Cleaned {len(text) - len(clean_text)} characters of markdown formatting for TTS")
        
        # Select TTS service URL
        tts_url = BARK_URL if tts_service == "bark" else ORPHEUS_URL
        tts_name = tts_service.capitalize()
        logger.info(f"Generating audio with {tts_name} TTS...")
        
        payload = {"text": clean_text, "voice": voice}
        
        response = requests.post(
            f"{tts_url}/synthesize",
            json=payload,
            timeout=600  # Increased to 10 minutes for long text generation
        )
        
        if response.status_code == 200:
            result = response.json()
            filename = result.get('filename')
            logger.info(f"Audio generated successfully with {tts_name}: {filename}")
            # Add URL for WebUI to access the audio
            return {
                "filename": filename,
                "url": f"http://localhost:8080/audio/{filename}"
            }
        elif response.status_code == 499:
            # TTS service was cancelled
            logger.info(f"TTS generation was cancelled on {tts_name} service")
            raise Exception("Task cancelled by user")
        else:
            logger.warning(f"Audio generation failed with {tts_name}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling {tts_service} TTS service: {e}")
        if "Task cancelled" in str(e):
            raise
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5009)
