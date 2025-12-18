# NISQA Service

Speech quality assessment service for TTS benchmark using NISQA-TTS model.

## Setup

```bash
cd services/modelNisqa

# Clone NISQA repository
git clone https://github.com/gabrielmittag/NISQA.git

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the service on port 8006:

```bash
./venv/bin/python app.py
```

## API

**Health Check:**
```bash
curl http://localhost:8006/health
```

**Predict MOS:**
```bash
curl -X POST http://localhost:8006/predict_from_path \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.wav"}'
```

**Response:**
```json
{
  "mos": 4.15,
  "latency": 0.234
}
```

Returns overall MOS score (1-5 scale) where higher is better.
