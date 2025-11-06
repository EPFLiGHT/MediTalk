# MediTalk - Medical AI with Voice

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

**MediTalk** is a research project that gives the Meditron medical AI model natural conversational speech capabilities, enabling voice-based medical consultations.

## Overview

MediTalk combines:
- **MultiMeditron** - Multimodal medical AI (text + audio context)
- **Text-to-Speech** - Orpheus TTS, Bark TTS, or CSM TTS
- **Speech Recognition** (Whisper ASR) - Voice input
- **Web Interface** - Interactive testing platform

### Complete Voice Pipeline
```
Voice/Text Input ──▶ Speech-to-Text ──▶ MultiMeditron ──▶ TTS ──▶ Voice Output
```

## Quick Start

### Prerequisites
### Prerequisites
- **Python 3.10+**
- **HuggingFace account** with token ([get one here](https://huggingface.co/settings/tokens))
- **16GB+ RAM** recommended for MultiMeditron

### 1. Configure Environment

Create a `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token_here # HuggingFace access token
MULTIMEDITRON_HF_TOKEN=your_token_here # For private MultiMeditron model
MULTIMEDITRON_MODEL=ClosedMeditron/Mulimeditron-End2End-CLIP-medical  # Multimodal AI
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft  # Medical TTS voice
WHISPER_MODEL=tiny # or base, small, medium, large
```

**Note:** You need access to:
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) (gated)
- [ClosedMeditron/Mulimeditron-End2End-CLIP-medical](https://huggingface.co/ClosedMeditron/Mulimeditron-End2End-CLIP-medical) (private)
  --> The private MultiMeditron model can be requested from EPFL LiGHT lab team.

## Deployment

**Setup (first run only):**
```bash
./scripts/setup-local.sh
```

**Start:**
```bash
./scripts/start-local.sh
```

**Monitor:**
```bash
# Check service health
./scripts/health-check.sh

# View logs
tail -f logs/modelMultiMeditron.log # Monitor MultiMeditron
tail -f logs/modelOrpheus.log # Monitor OrpheusTTS
tail -f logs/modelWhisper.log # Monitor Whisper
tail -f logs/webui.log # Monitor Web UI
```

**Stop:**
```bash
./scripts/stop-local.sh
```

## RCP Cluster Deployment

For deployment on the EPFL RCP cluster, please refer to the [LiGHT RCP Documentation](https://epflight.github.io/LiGHT-doc/clusters/rcp/) for setup instructions. After setting up the environment, you can clone this repository and follow the Deployment steps above to run MediTalk on the cluster.

## Services Architecture

| Service | Port | Purpose |
|---------|------|---------|
| **Web UI** | 8503 | Streamlit interactive interface |
| **Web UI API** | 8080 | FastAPI backend for audio/services |
| **MultiMeditron** | 5009 | Multimodal medical AI |
| **Orpheus TTS** | 5005 | Medical-focused text-to-speech |
| **Bark TTS** | 5008 | Multilingual text-to-speech |
| **CSM TTS** | 5010 | Conversational speech model |
| **Whisper ASR** | 5007 | Speech recognition |

## Model Configuration

### Speech Recognition
- **OpenAI Whisper**: Multiple sizes available (tiny, base, small, medium, large)

### AI Models
- **MultiMeditron**: Multimodal medical AI with audio context awareness (~8GB model + 8GB base LLM)

### TTS Models
- **Orpheus**: Medical-focused voice (best for medical terminology)
- **Bark**: Multilingual support with multiple voice presets
- **CSM (Sesame)**: Conversational speech model with context-based voice cloning

Switch models by editing `.env` and restarting services.

## API Usage

Notes: 
- We recommend using the Web UI for easier interaction. Below are example `curl` commands for direct API access.
- Make sure services are running before making requests.

### Medical AI Query
```bash
curl -X POST "http://localhost:5009/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is hypertension?",
    "generate_audio": true,
    "voice": "tara",
    "tts_service": "orpheus"
  }'
```

Parameters:
- `question`: Medical question to ask
- `generate_audio`: (bool) Whether to generate spoken response
- `voice`: Voice model to use for TTS (default: "tara", other voices may be available later)

### Text-to-Speech
```bash
curl -X POST "http://localhost:5005/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is MediTalk",
    "voice": "tara"
  }'
```

Parameters:
- `text`: Text to convert to speech
- `voice`: Voice model to use (default: "tara") 

### Speech-to-Text
```bash
curl -X POST "http://localhost:5007/transcribe" \
  -F "file=@audio.wav"
```
Parameters:
- `file`: Audio file to transcribe (WAV format recommended, other formats were not tested)

## Troubleshooting

### Local Deployment Issues

**Service won't start:**
```bash
# Check virtual environment
ls services/modelMultiMeditron/venv

# Reinstall dependencies
cd services/modelMultiMeditron
source venv/bin/activate
pip install -r requirements.txt
```

**Model loading errors:**
- Verify HuggingFace token in `.env`
- Check disk space (models are large)
- Review service logs in `logs/` directory

**Speech recognition issues:**
 
- ```bash
  # Ensure ffmpeg is installed:
  which ffmpeg

  # Install if missing (update package lists first to avoid 404 errors):
  sudo apt-get update
  sudo apt-get install -y ffmpeg
  
  # After installation, restart services to pick up ffmpeg in PATH:
  ./scripts/stop-local.sh
  ./scripts/start-local.sh
  ```
- Check audio file format (WAV recommended)
- Note: If you see "FileNotFoundError: 'ffmpeg'" in Whisper logs, the service needs to be restarted after ffmpeg installation to update its environment PATH

## Development

### Project Structure
```
MediTalk/
├── .env                    # Environment configuration
├── scripts/               # Shell scripts for service management
│   ├── setup-local.sh     # Creates virtual environments and installs dependencies
│   ├── start-local.sh     # Starts all services in background
│   ├── stop-local.sh      # Stops all running services
│   ├── restart.sh         # Restarts all services
│   ├── health-check.sh    # Checks service health status
│   └── monitor-gpus.sh    # GPU monitoring tool
├── services/
│   ├── modelMultiMeditron/ # Multimodal medical AI service
│   ├── modelOrpheus/      # Orpheus TTS service
│   ├── modelBark/         # Bark TTS service
│   ├── modelCSM/          # CSM TTS service (conversational)
│   ├── modelWhisper/      # ASR service
│   └── webui/             # Web interface (FastAPI + Streamlit)
├── logs/                  # Service logs
└── outputs/               # Generated audio and text files
```

### Key Scripts

All scripts are located in the `scripts/` folder:

- `setup-local.sh` - Creates virtual environments and installs dependencies
- `start-local.sh` - Starts all services in background
- `stop-local.sh` - Stops all running services
- `restart.sh` - Restarts all services
- `health-check.sh` - Checks service health status
- `monitor-gpus.sh` - Real-time GPU monitoring
```

### Key Files

- `monitor-gpus.sh` - Real-time GPU monitoring


## Research Context

### Current Phase: Baseline TTS Evaluation


## Research Context

### Current Phase: Baseline TTS Evaluation
This implementation tests existing TTS models (Orpheus, Bark, CSM) with MultiMeditron to establish performance benchmarks.

### Future Phase: Conversational Speech AI
Goal: Fine-tune Qwen model for end-to-end conversational speech generation with medical context awareness.

## Acknowledgments

### Models and Frameworks
- **[MultiMeditron](https://huggingface.co/ClosedMeditron/Mulimeditron-End2End-CLIP-medical)** - Multimodal medical language model by EPFL
- **[Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)** - Advanced neural text-to-speech
- **[Bark TTS](https://github.com/suno-ai/bark)** - Multilingual text-to-speech with voice presets
- **[OpenAI Whisper](https://huggingface.co/openai/whisper)** - Speech recognition model

### Technical Infrastructure
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern API framework
- **[Streamlit](https://streamlit.io/)** - Interactive web UI framework
- **[Hugging Face](https://huggingface.co/)** - Model hosting and distribution

### Future Research
- **[Qwen](https://github.com/QwenLM/Qwen)** - Target model for conversational speech fine-tuning

---

---

**MediTalk** - Empowering medical AI with natural conversational speech.

*Research Project | Nicolas Teissier | EPFL*
