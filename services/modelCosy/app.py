from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import tempfile
import torch
import sys

sys.path.append("CosyVoice")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

app = FastAPI(title="CosyVoice TTS Service")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "pretrained_models/CosyVoice2-0.5B"
cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

@app.get("/synthesize")
def synthesize(text: str = Query(..., min_length=1)):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    for i, j in enumerate(cosyvoice.inference_instruct2(text, '', '', stream=False)):
        torchaudio.save(tmp_file.name, j["tts_speech"], cosyvoice.sample_rate)
    return FileResponse(tmp_file.name, media_type="audio/wav")