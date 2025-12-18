# MediTalk - Medical AI with Voice

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

Medical conversational AI system combining MultiMeditron LLM with speech capabilities for voice-based medical interactions.

## Overview

MediTalk integrates multiple AI services to enable natural voice conversations with medical language models:

- **Medical LLM**: MultiMeditron for medical question answering
- **Speech Recognition**: Whisper for audio transcription
- **Speech Synthesis**: Multiple TTS models (Orpheus, Bark, CSM, Qwen3-Omni)
- **Web Interface**: Streamlit-based conversation UI
- **Benchmarking**: Comprehensive evaluation suite for TTS and ASR models

## Prerequisites

- Python 3.10+
- 48GB+ RAM
- HuggingFace token ([get one](https://huggingface.co/settings/tokens))
- Model access:
  - [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - ClosedMeditron/Mulimeditron-End2End-CLIP-medical (request from EPFL LiGHT lab)
  - [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)

## Quick Start

**1. Configure environment:**

Create `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token
MULTIMEDITRON_HF_TOKEN=your_token
MULTIMEDITRON_MODEL=ClosedMeditron/Mulimeditron-End2End-CLIP-medical
```

**2. Setup (first time only):**
```bash
./scripts/setup-local.sh
```

**3. Start services:**
```bash
./scripts/start-local.sh
```

**4. Access interface:**

Open http://localhost:8503 in your browser.

**5. Stop services:**
```bash
./scripts/stop-local.sh
```

## Services

MediTalk consists of multiple microservices. Each service has its own README with detailed setup instructions for individual API usage.

| Service | Port | Description | README |
|---------|------|-------------|--------|
| Controller | 8000 | Orchestrates all services | [Link](services/controller/README.md) |
| WebUI | 8501 | Streamlit interface | [Link](services/webui/README.md) |
| MultiMeditron | 5003 | Medical LLM | [Link](services/modelMultiMeditron/README.md) |
| Whisper | 8000 | Speech-to-text | [Link](services/modelWhisper/README.md) |
| Orpheus | 5005 | Neural TTS | [Link](services/modelOrpheus/README.md) |
| Bark | 5008 | Multilingual TTS | [Link](services/modelBark/README.md) |
| CSM | 5004 | Conversational TTS | [Link](services/modelCSM/README.md) |
| Qwen3-Omni | 5006 | Multimodal TTS | [Link](services/modelQwen3Omni/README.md) |
| NISQA | 8006 | Speech quality assessment | [Link](services/modelNisqa/README.md) |

## Benchmarking

MediTalk includes comprehensive benchmarking suites for evaluating model performance.

### TTS Benchmark

Evaluate text-to-speech models on intelligibility, quality, and speed.

```bash
cd benchmark/tts
./run_benchmark.sh
```

See [benchmark/tts/README.md](benchmark/tts/README.md) for details.

### Whisper Benchmark

Evaluate speech recognition accuracy across different Whisper model sizes.

```bash
cd benchmark/whisper
./run_benchmark.sh
```

See [benchmark/whisper/README.md](benchmark/whisper/README.md) for details.

## Project Structure

```
MediTalk/
│
├── services/                 # Microservices
│   ├── controller/           # Service orchestration
│   ├── webui/                # Web interface
│   ├── modelMultiMeditron/   # Medical LLM
│   ├── modelWhisper/         # ASR
│   ├── modelOrpheus/         # TTS
│   ├── modelBark/            # TTS
│   ├── modelCSM/             # TTS (conversational)
│   ├── modelQwen3Omni/       # TTS (conversational)
│   └── modelNisqa/           # Quality assessment (MOS)
│
├── benchmark/                # Evaluation suites
│   ├── tts/                  # TTS benchmark
│   └── whisper/              # ASR benchmark
│
├── scripts/                  # Management scripts
│
├── data/                     # Datasets (Download, Storage, Preprocessing) 
│
├── inputs/                   # Input files
│
├── outputs/                  # Generated files
│
└── logs/                     # Service logs
```

## Monitoring

**Check service health:**
```bash
./scripts/health-check.sh
```

**View logs:**
```bash
tail -f logs/controller.log
tail -f logs/modelOrpheus.log
```

**Monitor GPU usage:**
```bash
./scripts/monitor-gpus.sh
```

## Troubleshooting

**Service won't start:**
```bash
tail -f logs/<service>.log
```
Check for errors, missing dependencies, or invalid tokens.

**Missing ffmpeg:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
./scripts/restart.sh
```

**Model loading fails:**
- Verify HuggingFace token in `.env`
- Check disk space (models are large)
- Review service logs in `logs/` directory

Note: First run may take several minutes as models are downloaded and cached.

## EPFL RCP Cluster Deployment

For deployment on EPFL RCP cluster, refer to [LiGHT RCP Documentation](https://epflight.github.io/LiGHT-doc/clusters/rcp/).

## Acknowledgments

- [MultiMeditron](https://github.com/EPFLiGHT/MultiMeditron) - EPFL LiGHT Lab
- [Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) - Canopy Labs
- [Bark](https://github.com/suno-ai/bark) - Suno AI
- [Whisper](https://github.com/openai/whisper) - OpenAI
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) - Alibaba Cloud
- [NISQA](https://github.com/gabrielmittag/NISQA) - TU Berlin

---

*Semester Project | Nicolas Teissier | LiGHT Laboratory | EPFL*
