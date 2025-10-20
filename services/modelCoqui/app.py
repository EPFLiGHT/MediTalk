from fastapi import FastAPI
from pydantic import BaseModel
from TTS.api import TTS
import soundfile as sf
import time

app = FastAPI()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

class Req(BaseModel):
    text: str
    language: str = "en"

@app.post("/synthesize")
async def synth(req: Req):
    start = time.time()
    out_path = "/tmp/out.wav"
    tts.tts_to_file(text=req.text, speaker_wav=None, language=req.language, file_path=out_path)
    latency = time.time() - start
    duration = sf.info(out_path).duration
    return {"audio_path": out_path, "latency": latency, "duration": duration}