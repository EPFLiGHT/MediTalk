# CSM Service

Conversational speech synthesis using CSM (Conversational Speech Model) with context awareness.

## Setup

```bash
cd services/modelCSM

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the service on port 5004:

```bash
./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:5004/health
```

**Generate conversational speech:**
```bash
curl -X POST http://localhost:5004/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_path": "/path/to/conversation.json",
    "speaker": 0,
    "output_filename": "output.wav"
  }'
```

**Response:**
```json
{
  "audio_file": "/path/to/output.wav",
  "duration": 3.2,
  "generation_time": 2.1
}
```
