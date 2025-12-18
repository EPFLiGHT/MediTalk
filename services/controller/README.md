# Controller Service

Orchestration service for MediTalk medical conversation AI, coordinating between LLM, STT, and TTS services.

## Setup

```bash
cd services/controller

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

## API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Create conversation:**
```bash
curl -X POST http://localhost:8000/conversations
```

**Chat (text):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "message": "What are the symptoms of diabetes?"
  }'
```

**Transcribe & respond:**
```bash
curl -X POST http://localhost:8000/transcribe_and_chat \
  -F "audio=@input.wav" \
  -F "conversation_id=conv_123"
```

**Response:**
```json
{
  "conversation_id": "conv_123",
  "user_message": "What are the symptoms?",
  "assistant_message": "Common symptoms include...",
  "audio_file": "/path/to/response.wav"
}
```

The controller manages conversation state and coordinates between MultiMeditron (LLM), Whisper (STT), and TTS services.
