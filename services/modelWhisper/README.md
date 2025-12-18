# Whisper Service

Automatic speech recognition service using OpenAI Whisper for transcription.

## Setup

```bash
cd services/modelWhisper

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the service on port 8000:

```bash
./venv/bin/python app.py
```

Set model size via environment variable (default: base):
```bash
WHISPER_MODEL=medium ./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Transcribe from file path:**
```bash
curl -X POST http://localhost:8000/transcribe_from_path \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.wav"}'
```

**Response:**
```json
{
  "text": "transcribed text",
  "detected_language": "en",
  "latency": 0.523
}
```

Supports models: tiny, base, small, medium, large.
