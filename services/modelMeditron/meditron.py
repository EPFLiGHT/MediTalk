import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class MeditronLLM:
    def __init__(self, model_name=None, device=None):
        # Allow model override via environment variable
        if model_name is None:
            model_name = os.getenv('MEDITRON_MODEL', 'epfl-llm/meditron-7b')
        
        logger.info(f"Initializing {model_name} model...")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Handle Hugging Face authentication
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        try:
            logger.info("Loading Meditron tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=hf_token,
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading Meditron model...")
            
            # Memory-optimized loading for large models
            if "meditron-7b" in model_name.lower():
                logger.info("Using memory optimization for Meditron-7B...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device != "cpu" else None,
                    token=hf_token,
                    trust_remote_code=True,
                    offload_folder="./offload" if self.device == "cpu" else None,
                    load_in_8bit=False,  # Can be enabled if transformers supports it
                )
            else:
                # Lightweight model loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    token=hf_token,
                    trust_remote_code=True
                )
            
            # Move to device if not using device_map
            if not ("meditron-7b" in model_name.lower() and self.device != "cpu"):
                self.model = self.model.to(self.device)
            
            logger.info("Meditron-7B model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load Meditron model: {e}")
            raise e
    
    def generate_response(self, question: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate medical response using Meditron-7B"""
        try:
            # Format the prompt for medical context
            prompt = f"""As a medical AI assistant, please provide a helpful and accurate response to the following medical question:

Question: {question}

Answer:"""
            
            logger.info(f"Generating response for: '{question[:50]}...'")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part after "Answer:"
            if "Answer:" in full_response:
                response = full_response.split("Answer:")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            logger.info(f"Generated response ({len(response)} chars): {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise e