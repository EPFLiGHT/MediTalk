import torch
import soundfile as sf
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
import os
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)

# Set PyTorch CUDA memory allocation config to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
                    device_map="auto",  # Automatically manage device placement
                    token=hf_token  # Pass token explicitly
                )
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
        """
        Generate TTS audio from text and save to output_path.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            output_path: Path to save audio file
        """
        try:
            # Clear GPU cache before synthesis to free fragmented memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Split text into chunks for better quality
            logger.info(f"Synthesizing text (length={len(text)} chars) with voice '{voice}'")
            chunks = self._chunk_text(text, max_chars=250)
            logger.info(f"Text split into {len(chunks)} chunks")

            # Synthesize each chunk sequentially and collect audio arrays
            audio_pieces = []
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Synthesizing chunk {i+1}/{len(chunks)} (len={len(chunk)} chars)")
                prompt = f"{voice}: {chunk}"

                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

                input_ids = modified_input.to(self.device)
                attention_mask = torch.ones_like(input_ids)

                logger.info("Generating speech tokens for chunk...")
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10000,
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
                    logger.warning("No valid audio tokens generated for this chunk; skipping")
                    continue

                code_list = [(t - 128266).item() for t in cropped]

                # Decode to audio for this chunk
                logger.info("Decoding chunk to waveform...")
                audio = self._decode_snac(code_list)
                audio_numpy = audio.detach().squeeze().to("cpu").numpy()
                audio_pieces.append(audio_numpy)

            if len(audio_pieces) == 0:
                raise ValueError("No audio was generated for any chunk")

            # Concatenate pieces with a tiny fade to avoid clicks
            logger.info("Concatenating audio pieces...")
            final_audio = self._concatenate_audio(audio_pieces, sample_rate=24000)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            sf.write(output_path, final_audio, 24000)
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
        # Move tensors to the same device as the model
        codes = [torch.tensor(layer_1).unsqueeze(0).to(self.device),
                torch.tensor(layer_2).unsqueeze(0).to(self.device),
                torch.tensor(layer_3).unsqueeze(0).to(self.device)]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def _chunk_text(self, text, max_chars=400):
        """Split text into sentence-aware chunks not exceeding max_chars.

        Tries to split on sentence boundaries (., ?, !) and falls back to whitespace splitting.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return [text]

        sentences = re.split(r'(?<=[\.\?!])\s+', text)
        chunks = []
        current = []
        current_len = 0
        for s in sentences:
            s_len = len(s)
            if current_len + s_len + (1 if current_len>0 else 0) <= max_chars:
                current.append(s)
                current_len += s_len + (1 if current_len>0 else 0)
            else:
                if current:
                    chunks.append(' '.join(current))
                # If single sentence is longer than max_chars, we must split it
                if s_len > max_chars:
                    # fallback: split by words
                    words = s.split(' ')
                    part = []
                    part_len = 0
                    for w in words:
                        if part_len + len(w) + (1 if part_len>0 else 0) <= max_chars:
                            part.append(w)
                            part_len += len(w) + (1 if part_len>0 else 0)
                        else:
                            chunks.append(' '.join(part))
                            part = [w]
                            part_len = len(w)
                    if part:
                        chunks.append(' '.join(part))
                    current = []
                    current_len = 0
                else:
                    current = [s]
                    current_len = s_len

        if current:
            chunks.append(' '.join(current))
        return chunks

    def _concatenate_audio(self, pieces, sample_rate=24000):
        """Concatenate numpy audio arrays and apply short crossfade between pieces to avoid clicks."""
        if len(pieces) == 1:
            return pieces[0]

        # Determine fade length (in samples) - 10ms
        fade_ms = 10
        fade_len = int(sample_rate * (fade_ms/1000.0))

        out = pieces[0].copy()
        for p in pieces[1:]:
            # apply crossfade
            if fade_len > 0 and len(out) >= fade_len and len(p) >= fade_len:
                tail = out[-fade_len:]
                head = p[:fade_len]
                # linear fade
                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)
                cross = tail * fade_out + head * fade_in
                out = np.concatenate([out[:-fade_len], cross, p[fade_len:]])
            else:
                out = np.concatenate([out, p])
        return out