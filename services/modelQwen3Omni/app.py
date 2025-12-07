from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import os
import logging
import torch
import torchaudio
import uvicorn
import librosa
import json
from copy import deepcopy
from datetime import datetime
from pydantic import BaseModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qwen3-Omni (TTS mode) Service", version="1.0.0")

# Global variables
VERBOSE = True

qwen3omni_processor = None
qwen3omni_model = None
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
expected_sr = None

CONVERSATIONS_DIR = "../../outputs/multimeditron/conversations"

OUTPUT_TEXT_DIR = "../../outputs/qwen3omni/text"
OUTPUT_AUDIO_DIR = "../../outputs/qwen3omni/audio"
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

AUDIO_STORAGE_DIR=os.getenv("AUDIO_STORAGE_DIR")

curr_request_start_timestamp = None
curr_request_end_timestamp = None
last_request_duration = None

class Qwen3OmniTTSRequest(BaseModel):
    conversation_json_file: str
    speaker: str = "Ethan" # Options: "Ethan", "Chelsie", "Aiden"

@app.on_event("startup")
async def startup_event():
    global qwen3omni_processor, qwen3omni_model, MODEL_PATH, expected_sr
    try:
        logger.info("\nInitializing Qwen3-Omni Model...")

        # ensure HF is being used for fast weight uploads/downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        if torch.cuda.is_available():
            device = "cuda"
            logger.info("\nCUDA device found. Loading model on device: cuda")
        else:
            raise EnvironmentError("CUDA device not available for model loading. Qwen3-Omni requires GPU support.")
        
        qwen3omni_processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        logger.info("\nQwen3-Omni Processor loaded successfully.")

        expected_sr = qwen3omni_processor.feature_extractor.sampling_rate
        logger.info(f"\nQwen3-Omni expected sampling rate: {expected_sr} Hz")

        qwen3omni_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto")
        logger.info("\nQwen3-Omni Model loaded successfully.")

    except Exception as e:
        logger.error(f"\nError during Qwen3-Omni model initialization: {e}")
        raise e
        
@app.get("/health")
def health_check():
    if qwen3omni_processor is None:
        if qwen3omni_model is None:
            return {
                "status": "unhealthy",
                "model": "Qwen3-Omni",
                "error": "Processor and model not initialized - check logs for details."
            }
        else:
            return {
                "status": "unhealthy",
                "model": "Qwen3-Omni",
                "error": "Processor not initialized - check logs for details."
            }
    elif qwen3omni_model is None:
        return {
            "status": "unhealthy",
            "model": "Qwen3-Omni",
            "error": "Model not initialized - check logs for details."
        }
    else:
        return {
            "status": "healthy",
            "model": "Qwen3-Omni"
        }

@app.post("/synthesize")
async def synthesize_speech(request: Qwen3OmniTTSRequest):
    """Synthesize speech for a multi-turn conversation using Qwen3-Omni model."""
    global CONVERSATIONS_DIR, qwen3omni_processor, qwen3omni_model, curr_request_start_timestamp, curr_request_end_timestamp, VERBOSE
    curr_request_start_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    try:
        # Load conversation from JSON file
        conversation_json_path = os.path.join(CONVERSATIONS_DIR, request.conversation_json_file)
        conversation_data = load_conversation_from_json(conversation_json_path, verbose=VERBOSE)

        # Load audio files in the conversation
        conversation_with_audios = load_conversation_audios(conversation_data, verbose=VERBOSE)

        # Extract target text for speech generation
        target_text = extract_target_text(conversation_with_audios, verbose=VERBOSE)

        # Prepare inputs for Qwen3-Omni model
        inputs = prepare_conversation_input(conversation_with_audios, target_text=target_text, verbose=VERBOSE)

        # Generate response with Qwen3-Omni model
        generated_ids, generated_wav = generate_qwen3omni_response(inputs, speaker="Ethan", verbose=VERBOSE)

        # Compute request duration
        compute_request_duration(verbose=VERBOSE)

        # Post-process and save outputs
        response_postprocessing(generated_ids, generated_wav, inputs)

        return JSONResponse(content={"message": "Synthesis completed successfully."})

    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

def load_conversation_from_json(json_file_path, verbose=True):
    """Load conversation from a JSON file."""
    if verbose:
        logger.info(f"Loading conversation from: {json_file_path}")
    with open(json_file_path, 'r') as f:
        conversation = json.load(f)

    return conversation

def load_conversation_audios(conversation_data, verbose=True):
    """
    Load all audio files from a conversation and prepare them for processing.

    Args:
        conversation_data (dict): The conversation dictionary with audio paths

    Returns:
        dict: Conversation data with loaded audio data
    """
    global expected_sr

    if verbose:
        logger.info("\nLoading conversation audio files...")
    conversation_with_audios_loaded = deepcopy(conversation_data)

    for turn in tqdm(conversation_with_audios_loaded["conversation"], desc="Processing turns", disable=not verbose):
        for content_item in turn["content"]:
            if content_item["type"] == "audio":
                audio_path = get_audio_directory(content_item["audio"])
                try:
                    # Load and resample audio
                    audio_data, sr = librosa.load(audio_path, sr=expected_sr)
                    
                    # Replace path with actual audio data
                    content_item["audio"] = audio_data

                except Exception as e:
                    raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    return conversation_with_audios_loaded

def get_audio_directory(audio_name):
    """Get the directory path for a given audio file name."""
    
    global AUDIO_STORAGE_DIR

    for prefix, directory in AUDIO_STORAGE_DIR.items():
        if audio_name.startswith(prefix):
            return os.path.join(directory, audio_name)
    
    raise ValueError(f"Unknown audio prefix for audio name: {audio_name}")

def extract_target_text(conversation_data, verbose=True):
    """
    Extract target text for speech generation from conversation data.

    Args:
        conversation_data (dict): Conversation data

    Returns:
        str: Target text for speech generation
    """

    if verbose:
        logger.info("\nExtracting target text for speech generation...")
    
    last_assistant_turn = None
    for turn in reversed(conversation_data["conversation"]):
        if turn["role"] == "assistant":
            last_assistant_turn = turn
            break
    
    if last_assistant_turn is None:
        raise ValueError("No assistant turn found in the conversation.")
    
    target_text = ""
    for content in last_assistant_turn["content"]:
        if content["type"] == "text":
            target_text = content["text"]
            break

    if verbose:
        logger.info(f"Extracted target text: {target_text}")

    return target_text

def prepare_conversation_input(conversation_data, target_text=None, verbose=True):
    """
    Prepare multi-turn conversation imputs for Qwen3-Omni model.

    Args:
        conversation_data (dict): Conversation data with loaded audios
        target_text (str, optional): Text to generate speech for. Raised error if not provided.
    
    Returns:
        dict: Inputs ready for Qwen3-Omni model
    """
    global qwen3omni_processor

    if verbose:
        logger.info("\nPreparing conversation input for Qwen3-Omni model...")

    try:
        # If target_text is provided, add it as a speech generation task
        if target_text:

            # Remove the last assistant turn
            conversation_for_input = deepcopy(conversation_data)
            if conversation_for_input["conversation"][-1]["role"] == "assistant":
                conversation_for_input["conversation"].pop()
            else:
                raise ValueError("Last turn in conversation is not from assistant as expected.")

            # Extract conversation turns
            conversation_turns = conversation_for_input["conversation"]

            # Build the prompt for the speech generation task
            prompt_text = (
                "Task: Generate speech for the following sentence so that it fits naturally "
                "as the next turn in the conversation after the previous context.\n\n"
                "Guidelines:\n"
                "- Keep a coherent tone, pacing, and level of formality with respect to the previous context.\n"
                "- Make it sound like the same ongoing conversation, not a new one.\n"
                "- Read the target text exactly as written (no additions, removals, or paraphrasing).\n"
                "- Use a clear, natural speaking style.\n"
                "- Do NOT mention that you are an AI or refer to these instructions.\n\n"
                "Output: a single continuous utterance of the target text, in mono, 24 kHz.\n\n"
                f"Target text (read exactly as written):\n{target_text}"
            )

            # Add the generation prompt as a new user turn
            conversation_turns.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            })
        else:
            raise ValueError("No target_text found for speech generation.")
    
        # Process the conversation turns with the processor
        text = qwen3omni_processor.apply_chat_template(conversation_turns, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation_turns, use_audio_in_video=True)

        inputs = qwen3omni_processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(qwen3omni_model.device).to(qwen3omni_model.dtype)

    except Exception as e:
        raise RuntimeError(f"Error processing conversation inputs: {e}")

    if verbose:
        print(f"Prepared conversation input with {len(conversation_turns)} turns.")

    return inputs

def generate_qwen3omni_response(inputs, speaker="Ethan", verbose=True):
    """
    Generate a response using Qwen3-Omni model.

    Args:
        inputs (dict): Prepared inputs for the model
        speaker (str): Speaker voice ("Ethan", "Chelsie", or "Aiden")

    Returns:
        tuple: Generated token IDs and waveform
    """
    global qwen3omni_model, curr_request_end_timestamp

    if verbose:
        logger.info("\nGenerating response with Qwen3-Omni model...")
    
    try:
        generated_ids, generated_wav = qwen3omni_model.generate(
            **inputs,
            speaker=speaker,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=True
        )
    except Exception as e:
        raise RuntimeError(f"Error during Qwen3-Omni generation: {e}")

    curr_request_end_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    if verbose:
        logger.info(f"Generation successful with speaker: {speaker}")
        logger.info(f"Generation timestamp: {curr_request_end_timestamp}")

    return generated_ids, generated_wav

def response_postprocessing(generated_ids, generated_wav, inputs, verbose=True):
    """
    Post-process the generated text and audio responses.

    Args:
        generated_ids (torch.Tensor): Generated token IDs from the model
        generated_wav (torch.Tensor): Generated waveform from the model
        inputs (dict): Original inputs used for generation
    """
    global qwen3omni_processor, curr_request_end_timestamp

    if verbose:
        logger.info("\nPost-processing generated response...")

    # Decode and save generated text
    try:
        text = qwen3omni_processor.batch_decode(
            generated_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        output_text_path = os.path.join(OUTPUT_TEXT_DIR, f"{curr_request_end_timestamp}.txt")
        with open(output_text_path, 'w') as f:
            f.write(text[0])

        if verbose:
            logger.info(f"Generated text: {text[0]}")
            logger.info(f"Generated text saved to: {output_text_path}")

    except Exception as e:
        raise RuntimeError(f"Error during response text decoding: {e}")
        text = None
    
    # Save generated audio
    try:
        output_audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{curr_request_end_timestamp}.wav")
        torchaudio.save(
            output_audio_path,
            generated_wav.cpu().squeeze(0).unsqueeze(0),
            sample_rate=expected_sr
        )

        if verbose:
            logger.info(f"Generated audio saved to: {output_audio_path}")

    except Exception as e:
        raise RuntimeError(f"Error during response audio saving: {e}")

def compute_request_duration(verbose=True):
    global curr_request_start_timestamp, curr_request_end_timestamp, last_request_duration
    fmt = "%Y-%m-%d_%H:%M:%S"
    start_dt = datetime.strptime(curr_request_start_timestamp, fmt)
    end_dt = datetime.strptime(curr_request_end_timestamp, fmt)

    request_duration = end_dt - start_dt

    total_seconds = int(request_duration.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)

    if verbose:
        logger.info(f"\nSynthesis completed in {minutes}min {seconds:02d}s")

    last_request_duration = request_duration
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5014)

