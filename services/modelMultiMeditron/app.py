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


# Create output directory for text responses
OUTPUT_DIR = Path("/mloscratch/users/teissier/MediTalk/outputs/multimeditron")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Global model instance (following MultiMeditron README)
model = None
tokenizer = None
collator = None
image_loader = None
ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
attachment_token_idx = None

# Data models for backward compatibility with old /ask endpoint (if needed later)
class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class GenerateRequest(BaseModel):
    """Request from controller with conversation file path."""
    conversation_path: str
    max_length: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """Simple response with generated text only."""
    response: str


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
        
        # From here: the code follows the official documentation: https://epflight.github.io/MultiMeditron/guides/quickstart.html
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
        
        # Load MultiMeditron model
        logger.info(f"Loading MultiMeditron model from {model_path}...")
        from multimeditron.model.model import MultiModalModelForCausalLM
        from multimeditron.dataset.loader import FileSystemImageLoader
        from multimeditron.model.data_loader import DataCollatorForMultimodal
        
        model = MultiModalModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            token=hf_token
        )
        model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded with device_map='auto'")
        
        # Setup image loader
        image_loader = FileSystemImageLoader(base_path=os.getcwd())
        
        # Setup collator
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
        "attachment_token": ATTACHMENT_TOKEN
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate response from controller conversation file.
    
    Controller sends conversation JSON path. This endpoint:
    1. Reads the conversation JSON (controller format)
    2. Extracts last user message as the question
    3. Uses up to 10 previous messages as context
    4. Generates response
    5. Returns text only (controller handles conversation updates)
    """
    try:
        if not model or not tokenizer or not collator:
            raise HTTPException(status_code=503, detail="MultiMeditron model is not loaded")
        
        logger.info(f"Reading conversation from: {request.conversation_path}")
        
        # Read controller's conversation JSON
        with open(request.conversation_path, 'r') as f:
            conversation_data = json.load(f)
        
        messages = conversation_data.get('messages', [])
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages in conversation")
        
        # Find last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                last_user_message = msg
                break
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found in conversation")
        
        # Extract text from last user message
        question = ""
        for content_item in last_user_message.get('content', []):
            if content_item.get('type') == 'text':
                question = content_item.get('data', '')
                break
        
        if not question:
            raise HTTPException(status_code=400, detail="No text content in last user message")
        
        logger.info(f"Extracted question: {question[:100]}...")
        
        # Build conversation history (up to 10 previous messages)
        # Exclude the last user message (it's the question)
        history_messages = messages[:-1] if len(messages) > 1 else []
        
        # Limit to last 10 messages
        if len(history_messages) > 10:
            history_messages = history_messages[-10:]
        
        # Convert to old ConversationMessage format for generate_multimeditron_response
        conversation_history = []
        for msg in history_messages:
            # Extract text content
            text_content = ""
            for content_item in msg.get('content', []):
                if content_item.get('type') == 'text':
                    text_content = content_item.get('data', '')
                    break
            
            if text_content:
                conversation_history.append(
                    ConversationMessage(
                        role=msg.get('role'),
                        content=text_content
                    )
                )
        
        logger.info(f"Using {len(conversation_history)} previous messages as context")
        
        # Generate response using existing function
        answer = generate_multimeditron_response(
            question=question,
            modalities=[],  # No multimodal for now
            temperature=request.temperature,
            max_length=request.max_length,
            conversation_history=conversation_history
        )
        
        logger.info(f"Generated response: {answer[:100]}...")
        
        return GenerateResponse(response=answer)
        
    except FileNotFoundError:
        logger.error(f"Conversation file not found: {request.conversation_path}")
        raise HTTPException(status_code=404, detail="Conversation file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in conversation file: {e}")
        raise HTTPException(status_code=400, detail="Invalid conversation JSON format")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5009)
