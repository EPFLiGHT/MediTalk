from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Union
from pydantic import BaseModel, Field
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="MultiMeditron Multimodal Medical AI Service", version="1.0.0")


# Service URLs
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
BARK_URL = os.getenv("BARK_URL", "http://localhost:5008")
CSM_URL = os.getenv("CSM_URL", "http://localhost:5010")


# Create output directory for text responses
OUTPUT_DIR = Path("/mloscratch/users/teissier/MediTalk/outputs/multimeditron")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create directory for conversation history JSON files
CONVERSATION_DIR = OUTPUT_DIR / "conversations"
CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)


# Global model instance (following MultiMeditron README)
model = None
tokenizer = None
collator = None
image_loader = None
ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
attachment_token_idx = None

# Global conversation tracker
current_conversation = JsonConversationBuilder()

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

# New Data models for conversation history
class AudioContent(BaseModel):
    type: Literal["audio"]
    audio: str  # file path

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

ContentItem = Union[AudioContent, TextContent]

class ConversationMyMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: List[ContentItem] = Field(discriminator='type')

class JsonConversationBuilder(BaseModel):
    conversation: List[ConversationMyMessage] = []
    
    def add_turn(self, role: str, text: str, audio_path: Optional[str] = None) -> None:
        """Add a conversation turn with text and optional audio."""
        content_items = []
        
        if audio_path:
            content_items.append(AudioContent(type="audio", audio=audio_path))
        
        content_items.append(TextContent(type="text", text=text))
        
        message = ConversationMyMessage(role=role, content=content_items)
        self.conversation.append(message)
    
    def to_dict(self) -> dict:
        """Export conversation to dictionary format."""
        return {"conversation": [msg.model_dump() for msg in self.conversation]}
    
    def to_json(self, filepath: str) -> None:
        """Save conversation to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> "JsonConversationBuilder":
        """Load conversation from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(conversation=[ConversationMyMessage(**msg) for msg in data["conversation"]])

# Old data model for compatibility
class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    audio_url: Optional[str] = None  # Optional audio URL for CSM context

# Other non-specific data models
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 512
    temperature: float = 0.7
    generate_audio: bool = True
    voice: str = "tara"
    tts_service: str = "orpheus"  # "orpheus", "bark", or "csm"
    language: str = "en"  # Language for Orpheus TTS: "en" or "fr"
    conversation_history: List[ConversationMessage] = []  # For maintaining conversation context
    generate_in_parallel: bool = True  # For Orpheus TTS parallel processing

class MedicalResponse(BaseModel):
    question: str
    answer: str
    audio_file: Optional[str] = None
    audio_url: Optional[str] = None
    context_skipped: Optional[bool] = None  # Flag when context was too long

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
        
        # Load tokenizer
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
        
        logger.info("MultiMeditron fully loaded and ready!")
        
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
        "attachment_token": ATTACHMENT_TOKEN,
        "conversation_turns": len(current_conversation.conversation) if current_conversation else 0
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
            max_length=request.max_length,
            conversation_history=request.conversation_history
        )
        
        response_data = {
            "question": request.question,
            "answer": answer,
        }
        
        # Save text response for debugging
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_filename = f"response_{timestamp}.txt"
            text_path = OUTPUT_DIR / text_filename
            with open(text_path, "w") as f:
                f.write(f"QUESTION:\n{request.question}\n\n")
                f.write(f"ANSWER:\n{answer}\n\n")
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"TEMPERATURE: {request.temperature}\n")
                f.write(f"MAX_LENGTH: {request.max_length}\n")
            logger.info(f"Saved text response to {text_filename}")
        except Exception as e:
            logger.warning(f"Failed to save text response: {e}")
        
        # Generate audio if requested
        if request.generate_audio:
            try:
                logger.info(f"Generating audio with {request.tts_service}...")
                audio_result = generate_audio(
                    text=answer, 
                    voice=request.voice, 
                    tts_service=request.tts_service,
                    language=request.language,
                    conversation_history=request.conversation_history,
                    generate_in_parallel=request.generate_in_parallel
                )
                if audio_result:
                    response_data["audio_file"] = audio_result.get("filename")
                    response_data["audio_url"] = audio_result.get("url")
                    # Include context_skipped flag if present (for CSM)
                    if "context_skipped" in audio_result:
                        response_data["context_skipped"] = audio_result["context_skipped"]
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
    max_length: int = 512,
    conversation_history: List[ConversationMessage] = []
    ) -> str:
    """
    Generate response using MultiMeditron model
    Following the exact pattern from MultiMeditron README
    """
    global model, tokenizer, collator, attachment_token_idx
    
    # Build the conversations list with history + current question
    conversations = []
    
    # Add conversation history
    for msg in conversation_history:
        conversations.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add current user question
    conversations.append({
        "role": "user",
        "content": question
    })
    
    sample = {
        "conversations": conversations,
        "modalities": modalities
    }
    
    logger.info(f"Generating response for question: {question[:100]}")
    logger.info(f"With conversation history of {len(conversation_history)} messages")
    logger.info(f"With {len(modalities)} modalities")
    
    # Create batch using collator (following README)
    batch = collator([sample])
    
    # Generate using model (following README)
    with torch.no_grad():
        outputs = model.generate(
            batch=batch, 
            temperature=temperature,
            max_new_tokens=max_length
        )
    
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

def generate_audio(text: str, voice: str, tts_service: str = "orpheus", language: str = "en", conversation_history: List[ConversationMessage] = [], generate_in_parallel: bool = True) -> Optional[Dict]:
    """
    Generate audio using specified TTS service (Orpheus, Bark, or CSM)
    For CSM, conversation_history is used to provide context for more natural speech
    For Orpheus, generate_in_parallel controls dynamic multi-instance parallelization
    For Orpheus, language controls which model to use ("en" or "fr")
    
    Returns:
        {"filename": str, "url": str}
    """
    try:
        # Clean markdown formatting from text before TTS
        clean_text = clean_text_for_tts(text)
        logger.info(f"Cleaned {len(text) - len(clean_text)} characters of markdown formatting for TTS")
        
        # Handle CSM (Conversational Speech Model) - requires conversation context
        if tts_service == "csm":
            logger.info("Generating audio with CSM (conversational TTS)...")
            
            # Build conversation context for CSM
            context_segments = []
            for i, msg in enumerate(conversation_history[-10:]):  # Last 10 messages for context
                # CSM uses speaker IDs: 0 for one speaker, 1 for another
                # We'll use 0 for user, 1 for assistant
                speaker_id = 0 if msg.role == "user" else 1
                
                # Check if this message has associated audio
                audio_url = None
                if hasattr(msg, 'audio_url') and msg.audio_url:
                    audio_url = msg.audio_url
                
                context_segments.append({
                    "text": clean_text_for_tts(msg.content),
                    "speaker": speaker_id,
                    "audio_url": audio_url
                })
            
            # CSM needs to know which speaker is generating (assistant = 1)
            speaker_id = int(voice) if voice in ["0", "1"] else 1
            
            payload = {
                "text": clean_text,
                "speaker": speaker_id,
                "conversation_history": context_segments,
                "max_audio_length_ms": 10000  # Per chunk - CSM now uses chunking
            }
            
            response = requests.post(
                f"{CSM_URL}/synthesize",
                json=payload,
                timeout=600
            )
            
            if response.status_code == 200:
                result = response.json()
                filename = result.get('filename')
                context_skipped = result.get('context_skipped', False)
                logger.info(f"Audio generated successfully with CSM: {filename}")
                if context_skipped:
                    logger.warning(r"/!\ CSM context was skipped due to length constraints")
                return {
                    "filename": filename,
                    "url": f"http://localhost:8080/audio/{filename}",
                    "context_skipped": context_skipped
                }
            else:
                logger.warning(f"Audio generation failed with CSM: {response.status_code}")
                return None
        
        # Handle Orpheus or Bark (non-conversational TTS)
        else:
            # Determine TTS URL
            if tts_service == "orpheus":
                tts_url = ORPHEUS_URL
                tts_name = "Orpheus"
            else:
                tts_url = BARK_URL
                tts_name = "Bark"
            
            logger.info(f"Generating audio with {tts_name} TTS...")
            
            payload = {"text": clean_text, "voice": voice}
            
            # Add generate_in_parallel parameter for Orpheus
            if tts_service == "orpheus":
                payload["generate_in_parallel"] = generate_in_parallel
                payload["language"] = language
                logger.info(f"Orpheus parallel mode: {generate_in_parallel}, language: {language}")
            
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
