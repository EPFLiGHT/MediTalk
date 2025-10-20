import torch
import soundfile as sf
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
import os
import logging

logger = logging.getLogger(__name__)

class OrpheusTTS:
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-ft", device=None, fallback_model=None):
        logger.info("Initializing Orpheus model...")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Handle Hugging Face authentication
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            logger.info("Authenticating with Hugging Face...")
            try:
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")
        else:
            logger.warning("No HUGGINGFACE_TOKEN found. You may not be able to access gated models.")
            logger.info("To use Orpheus, you need to:")
            logger.info("   1. Get a Hugging Face token: https://huggingface.co/settings/tokens")
            logger.info("   2. Request access to: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft")
            logger.info("   3. Set HUGGINGFACE_TOKEN environment variable")

        # Try to load the model with error handling
        models_to_try = [model_name]
        if fallback_model:
            models_to_try.append(fallback_model)

        for model_to_load in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_to_load}")
                
                # Load SNAC decoder
                logger.info("Loading SNAC decoder...")
                self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)

                # Download model weights/config
                logger.info("Downloading Orpheus model from Hugging Face...")
                snapshot_download(
                    repo_id=model_to_load,
                    allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json"],
                    token=hf_token  # Pass token explicitly
                )

                logger.info("Loading Orpheus model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_to_load, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    token=hf_token  # Pass token explicitly
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, token=hf_token)
                
                logger.info(f"Successfully loaded model: {model_to_load}")
                self.current_model = model_to_load
                return
                
            except Exception as e:
                logger.error(f"Failed to load model {model_to_load}: {e}")
                if model_to_load == models_to_try[-1]:  # Last model in list
                    logger.error("All model loading attempts failed!")
                    raise e
                else:
                    logger.info(f"Trying next model...")
                    continue

    def synthesize(self, text, voice="tara", output_path="output.wav"):
        """Generate TTS audio from text and save to output_path."""
        try:
            prompt = f"{voice}: {text}"
            logger.info(f"Synthesizing: '{text}' with voice '{voice}'")

            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

            input_ids = modified_input.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            logger.info("Generating speech tokens...")
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=128258,
                )

            # Post-process tokens
            token_to_find = 128257
            token_to_remove = 128258
            indices = (generated == token_to_find).nonzero(as_tuple=True)
            cropped = generated[:, indices[1][-1]+1:] if len(indices[1]) > 0 else generated
            cropped = cropped[cropped != token_to_remove]
            
            if len(cropped) == 0:
                raise ValueError("No valid audio tokens generated")
                
            code_list = [(t - 128266).item() for t in cropped]

            # Decode to audio
            logger.info("Decoding to waveform...")
            audio = self._decode_snac(code_list)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to numpy array following official implementation
            audio_numpy = audio.detach().squeeze().to("cpu").numpy()
            sf.write(output_path, audio_numpy, 24000)
            logger.info(f"Audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise e

    def _decode_snac(self, code_list):
        """Official redistribution logic from orpheus_0_1_finetune_inference.py"""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        # Follow the official implementation exactly
        codes = [torch.tensor(layer_1).unsqueeze(0),
                torch.tensor(layer_2).unsqueeze(0),
                torch.tensor(layer_3).unsqueeze(0)]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat