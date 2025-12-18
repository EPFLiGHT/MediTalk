# Orpheus Service

Neural text-to-speech service using Canopy Labs Orpheus model with multilingual support.

## Setup

```bash
cd services/modelOrpheus

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Hugging Face token:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

Request access to models:
- https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
- https://huggingface.co/canopylabs/3b-fr-ft-research_release

## Usage

Start the service on port 5005:

```bash
./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:5005/health
```

**Generate speech:**
```bash
curl -X POST http://localhost:5005/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice": "tara",
    "language": "en",
    "output_filename": "output.wav"
  }'
```

**Response:**
```json
{
  "audio_file": "/path/to/output.wav",
  "duration": 2.5,
  "generation_time": 1.2
}
```

Supported voices: tara, nova, sage. Languages: en, fr.
