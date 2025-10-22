# MediTalk - Medical AI with Voice

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

**MediTalk** is a research project that gives the Meditron medical AI model natural conversational speech capabilities, enabling voice-based medical consultations.

## Overview

MediTalk combines:
- **Meditron-7B** - Text-only medical question answering
- **MultiMeditron** - Multimodal medical AI (text + images -> not supported yet)
- **Text-to-Speech** - Orpheus TTS or Bark TTS
- **Speech Recognition** (Whisper ASR) - Voice input
- **Web Interface** - Interactive testing platform

### Complete Voice Pipeline
```
Voice/Text Input ──▶ Speech-to-Text ──▶ MultiMeditron ──▶ TTS ──▶ Voice Output
```

## Quick Start

### Prerequisites
- **Python 3.10+** (for local) or **Docker** (for containerized)
- **HuggingFace account** with token ([get one here](https://huggingface.co/settings/tokens))
- **16GB+ RAM** recommended for Meditron-7B (or 8GB for DialoGPT)

### 1. Configure Environment

Create a `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token_here # HuggingFace access token
MULTIMEDITRON_HF_TOKEN=your_token_here # For private MultiMeditron model
MEDITRON_MODEL=epfl-llm/meditron-7b  # Text-only medical AI
MULTIMEDITRON_MODEL=ClosedMeditron/Mulimeditron-End2End-CLIP-medical  # Multimodal AI
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft  # Medical TTS voice
WHISPER_MODEL=tiny # or base, small, medium, large
```

**Note:** You need access to:
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) (gated)
- [ClosedMeditron/Mulimeditron-End2End-CLIP-medical](https://huggingface.co/ClosedMeditron/Mulimeditron-End2End-CLIP-medical) (private)
  --> The private MultiMeditron model can be requested from EPFL LiGHT lab team.

## Deployment Options

### Option A: Local Deployment (No Docker)

Note: This is the recommended option for the EPFL RCP cluster (see RCP Cluster Deployment section below).

**Setup (first run only):**
```bash
./setup-local.sh
```

**Start:**
```bash
./start-local.sh
```

**Monitor:**
```bash
# Check service health
./health-check.sh

# View logs
tail -f logs/modelMeditron.log # Monitor Meditron
tail -f logs/modelOrpheus.log # Monitor OrpheusTTS
tail -f logs/modelWhisper.log # Monitor Whisper
tail -f logs/webui.log # Monitor Web UI
```

**Stop:**
```bash
./stop-local.sh
```

### Option B: Docker Deployment

Note: We had reasons to believe this option may not work on the EPFL RCP cluster. We recommend using local deployment instead.

**Start:**
```bash
./start-docker.sh
```

**Monitor:**
```bash
# Check service status
docker compose -f docker/docker-compose.yml ps

# View logs
docker compose -f docker/docker-compose.yml logs -f meditron
docker compose -f docker/docker-compose.yml logs -f orpheus
docker compose -f docker/docker-compose.yml logs -f whisper
docker compose -f docker/docker-compose.yml logs -f webui

# Check health
curl http://localhost:5006/health  # Meditron
curl http://localhost:5005/health  # Orpheus TTS
curl http://localhost:5007/health  # Whisper ASR
```

**Stop:**
```bash
docker compose -f docker/docker-compose.yml down
```

## RCP Cluster Deployment

For deployment on the EPFL RCP cluster, please refer to the [LiGHT RCP Documentation](https://epflight.github.io/LiGHT-doc/clusters/rcp/) for setup instructions. After setting up the environment, you can clone this repository and follow the Local Deployment steps above to run MediTalk on the cluster.

Recommendation: Use option A (Local Deployment) as Docker deployment may not be supported on the RCP cluster.

## Services Architecture

| Service | Port | Purpose |
|---------|------|---------|
| **Web UI** | 8080 | Interactive interface with AI/TTS model selection |
| **Meditron** | 5006 | Text-only medical Q&A |
| **MultiMeditron** | 5009 | Multimodal medical AI |
| **Orpheus TTS** | 5005 | Text-to-speech model|
| **Bark TTS** | 5008 | Multilingual text-to-speech |
| **Whisper ASR** | 5007 | Speech recognition |

## Model Configuration

### Speech Recognition
- **OpenAI Whisper**: Multiple sizes available (tiny, base, small, medium, large)

### AI Models
- **DialoGPT**: Lightweight conversational model for initial testing (~345MB, 1-2GB RAM)
- **Meditron-7B**: Text-only medical Q&A (~14GB, 16GB+ RAM)
- **MultiMeditron**: Multimodal (text + images -> not supported yet) medical AI (~8GB model + 8GB base LLM)

### TTS Models
- **Orpheus**: Medical-focused voice (best for medical terminology)
- **Bark**: Multilingual support with multiple voice presets

Switch models by editing `.env` and restarting services or using the model selector in the Web UI.

## API Usage

Notes: 
- We recommend using the Web UI for easier interaction. Below are example `curl` commands for direct API access.
- Make sure services are running before making requests. (See Deployment Options above and Health Check)

### Medical AI Query
```bash
curl -X POST "http://localhost:5006/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is hypertension?",
    "generate_audio": true,
    "voice": "tara"
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
ls services/modelMeditron/venv

# Reinstall dependencies
cd services/modelMeditron
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
  ./stop-local.sh
  ./start-local.sh
  ```
- Check audio file format (WAV recommended)
- Note: If you see "FileNotFoundError: 'ffmpeg'" in Whisper logs, the service needs to be restarted after ffmpeg installation to update its environment PATH

### Docker Deployment Issues

**Memory errors (Exit Code 137):**
```bash
# This indicates out-of-memory issues.
# Switch to lighter model
MEDITRON_MODEL=microsoft/DialoGPT-medium
```

**Container won't start:**
```bash
# Check Docker resources
docker stats

# Rebuild containers
docker compose -f docker/docker-compose.yml build --no-cache
```

## Development

### Project Structure
```
MediTalk-NoDocker/
├── .env                    # Environment configuration
├── services/
│   ├── modelMeditron/     # Medical AI service
│   ├── modelOrpheus/      # TTS service
│   ├── modelWhisper/      # ASR service
│   └── webui/             # Web interface
├── docker/                # Docker configuration
├── logs/                  # Service logs (local mode)
├── outputs/               # Generated audio files
├── setup-local.sh         # Local setup script
├── start-local.sh         # Local start script
├── health-check.sh        # Local health check script
├── stop-local.sh          # Local stop script
└── start-docker.sh        # Docker start script
```

### Key Files

**Local Mode:**
- `setup-local.sh` - Creates virtual environments and installs dependencies
- `start-local.sh` - Starts all services in background
- `stop-local.sh` - Stops all running services
- `health-check.sh` - Checks service health status

**Docker Mode:**
- `start-docker.sh` - Builds and starts Docker containers
- `docker/docker-compose.yml` - Service orchestration
- `docker/Dockerfile.base` - Base image configuration (see individual service Dockerfiles for specifics on each service)


## Research Context

### Current Phase: Baseline TTS Evaluation
This implementation tests existing TTS models (Orpheus, other TTS models later) with Meditron to establish performance benchmarks.

### Future Phase: Conversational Speech AI
Goal: Fine-tune Qwen model for end-to-end conversational speech generation with medical context awareness.

## Acknowledgments

### Models and Frameworks
- **[DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium)** - Baseline conversational model
- **[Meditron](https://huggingface.co/epfl-llm/meditron-7b)** - Medical language model by EPFL
- **[Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)** - Advanced neural text-to-speech
- **[OpenAI Whisper](https://huggingface.co/openai/whisper)** - Speech recognition model

### Technical Infrastructure
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern API framework
- **[Docker](https://www.docker.com/)** - Containerization platform
- **[Hugging Face](https://huggingface.co/)** - Model hosting and distribution

### Future Research
- **[Qwen](https://github.com/QwenLM/Qwen)** - Target model for conversational speech fine-tuning

---

---

**MediTalk** - Empowering medical AI with natural conversational speech.

*Research Project | Nicolas Teissier | EPFL*
