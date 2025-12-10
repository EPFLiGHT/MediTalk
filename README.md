# MediTalk - Medical AI with Voice

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

**MediTalk** is a research project that gives the MultiMeditron medical LLM model from LiGHT Laboratory natural conversational speech capabilities, enabling voice-based medical interactions.

## Prerequisites

- Python 3.10+
- 48GB+ RAM
- HuggingFace token ([get one](https://huggingface.co/settings/tokens))
- Access to [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Access to ClosedMeditron/Mulimeditron-End2End-CLIP-medical (request from EPFL LiGHT lab)

## Setup

Create `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token
MULTIMEDITRON_HF_TOKEN=your_token
MULTIMEDITRON_MODEL=ClosedMeditron/Mulimeditron-End2End-CLIP-medical
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft
WHISPER_MODEL=base
```

## Deployment

**Setup environments (first run only):**
```bash
./scripts/setup-local.sh
```

**Start all services:**
```bash
./scripts/start-local.sh
```

**Access web interface:** `http://localhost:8503`

**Stop services:**
```bash
./scripts/stop-local.sh
```

**Monitor:**
```bash
./scripts/health-check.sh
tail -f logs/controller.log
```

## RCP Cluster Deployment

For deployment on the EPFL RCP cluster, please refer to the [LiGHT RCP Documentation](https://epflight.github.io/LiGHT-doc/clusters/rcp/) for setup instructions. After setting up the environment, you can clone this repository and follow the deployment steps above to run MediTalk on the cluster.

## Services Architecture

| Service | Port | Description |
|---------|------|-------------|
| Controller | 8000 | Orchestrates LLM, TTS, STT services |
| Web UI | 8503 | Streamlit interface |
| MultiMeditron | 5009 | Medical AI model |
| Whisper | 5007 | Speech-to-text |
| Orpheus | 5005 | Text-to-speech |
| Bark | 5008 | Text-to-speech |
| CSM | 5010 | Conversational Text-to-speech |
| Qwen3-Omni | 5014 | Conversational Text-to-speech |

## Troubleshooting

**Service won't start:**
```bash
tail -f logs/<service>.log
```
Check for errors, missing dependencies or missing tokens.

Note: Some services may take several minutes to load models on first run.

**Missing ffmpeg:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
./scripts/restart.sh
```

**Model loading fails:**
- Verify HuggingFace token in `.env`
- Check disk space (models are large)
- Review service logs in `logs/` directory

## Project Structure

```
MediTalk/
├── .env                      # Environment configuration
├── scripts/                  # Service management scripts
│   ├── setup-local.sh        # Install dependencies
│   ├── start-local.sh        # Start all services
│   ├── stop-local.sh         # Stop all services
│   ├── restart.sh            # Restart services
│   ├── health-check.sh       # Check service health
│   └── monitor-gpus.sh       # GPU monitoring
├── services/
│   ├── controller/           # Orchestration service
│   ├── modelMultiMeditron/   # Medical LLM
│   ├── modelWhisper/         # Speech-to-text
│   ├── modelOrpheus/         # TTS
│   ├── modelBark/            # TTS
│   ├── modelCSM/             # TTS (conversational)
│   ├── modelQwen3Omni/       # TTS (conversational)
│   └── webui/                # Streamlit interface
├── inputs/                   # Input files (conversations json files)
├── outputs/                  # Generated audio files
└── logs/                     # Service logs
```

## Acknowledgments

- [MultiMeditron] - EPFL LiGHT Lab
- [Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) - Canopy Labs
- [Bark](https://github.com/suno-ai/bark) - Suno AI
- [Whisper](https://github.com/openai/whisper) - OpenAI
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) - Alibaba Cloud
- [FastAPI](https://fastapi.tiangolo.com/) & [Streamlit](https://streamlit.io/)

---

*Semester Project | Nicolas Teissier | LiGHT Laboratory | EPFL*
