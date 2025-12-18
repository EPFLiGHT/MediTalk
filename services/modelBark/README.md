# Bark Service

Text-to-speech service using Suno AI Bark model with multilingual voice presets.

## Setup

```bash
cd services/modelBark

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the service on port 5008:

```bash
./venv/bin/python app.py
```

First run downloads models (approximately 5GB, may take 5-10 minutes).

## API

**Health Check:**
```bash
curl http://localhost:5008/health
```

**Generate speech:**
```bash
curl -X POST http://localhost:5008/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice": "v2/en_speaker_6",
    "output_filename": "output.wav"
  }'
```

**Response:**
```json
{
  "audio_file": "/path/to/output.wav",
  "duration": 2.3,
  "generation_time": 1.8
}
```

**List voices:**
```bash
curl http://localhost:5008/voices
```

Supports 10+ voices in multiple languages (en, es, fr, de, zh, etc.).

