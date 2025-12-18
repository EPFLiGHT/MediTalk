# MultiMeditron Service

Multimodal medical AI service using MultiMeditron for medical conversation generation.

## Setup

```bash
cd services/modelMultiMeditron

# Clone MultiMeditron repository
git clone https://github.com/epfml/MultiMeditron.git multimeditron

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ./multimeditron
```

Set your Hugging Face token:
```bash
export MULTIMEDITRON_HF_TOKEN=your_token_here
```

Note: Request this token for `ClosedMeditron/Mulimeditron-End2End-CLIP-medical` model from the LiGHT Laboratory at EPFL.

## Usage

Start the service on port 5003:

```bash
./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:5003/health
```

**Generate medical response:**
```bash
curl -X POST http://localhost:5003/generate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_path": "/path/to/conversation.json",
    "max_length": 512,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "response": "Generated medical response text"
}
```

MultiMeditron specializes in medical conversations with support for multimodal inputs.
