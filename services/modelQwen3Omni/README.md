# Qwen3-Omni Service

Multimodal conversational AI with text-to-speech generation using Qwen3-Omni model.

## Setup

```bash
cd services/modelQwen3Omni

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the service on port 5006:

```bash
./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:5006/health
```

**Generate speech:**
```bash
curl -X POST http://localhost:5006/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_path": "/path/to/conversation.json",
    "speaker": "Ethan",
    "output_filename": "output.wav"
  }'
```

**Response:**
```json
{
  "audio_file": "/path/to/output.wav",
  "duration": 4.1,
  "generation_time": 3.5
}
```

Supported speakers: Ethan, Chelsie, Aiden.
