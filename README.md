# MediTalk - Giving Meditron a Voice

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

**MediTalk** is a research project focused on giving the Meditron medical AI model natural conversational speech capabilities. This repository contains the baseline testing pipeline for evaluating various TTS approaches before developing the final solution: **fine-tuning Qwen for direct conversational speech generation**.

## Project Vision

### Research Goal
Create a medical AI that can engage in natural spoken conversations, moving beyond traditional text-to-speech to **end-to-end conversational speech generation**.

### Current Phase: Baseline TTS Testing
This implementation serves as a **testing framework** to evaluate existing TTS models with Meditron, establishing performance benchmarks before developing the advanced conversational speech system.

### Future Phase: Conversational Speech AI
The ultimate goal is to **fine-tune Qwen model for conversational speech generation**, where Meditron generates medical content and Qwen produces natural speech with appropriate conversational tone and context awareness.

## Architecture Evolution

### Target Conversational Pipeline
```
┌───────────────────┐     ┌───────────┐     ┌─────────────────┐     ┌──────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ User Speech Input │ ──▶ │ ASR Model │ ──▶ │ User Text Input │ ──▶ │ Meditron │ ──▶ │ Medical Response │ ──▶ │ Fine-tuned Qwen │ ──▶ │  Natural Speech │
└───────────────────┘     └───────────┘     └─────────────────┘     └──────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
                                │                                        │                                             │
                         [Speech-to-Text]                           [Medical AI]                              [Conversational TTS]
```

## Speech-to-Speech Capabilities

MediTalk now supports **full voice interaction**, enabling natural spoken conversations with the medical AI:

### Voice Input Options
- **Browser Web Speech API** (Recommended) - Real-time speech recognition using browser capabilities
- **OpenAI Whisper ASR** (Experimental) - Advanced speech-to-text model with high accuracy (may have compatibility issues on some hardware at the moment)

### Complete Voice Pipeline
```
┌──────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────────┐
│ Voice Input  │ ──▶ │   ASR    │ ──▶ │ Meditron │ ──▶ │ Orpheus TTS │ ──▶ │ Voice Output │
└──────────────┘     └──────────┘     └──────────┘     └─────────────┘     └──────────────┘
       │                   │                 │                  │                   │
   [Speak]           [Transcribe]      [Generate]         [Synthesize]         [Listen]
```

## TTS Model Testing Framework

This baseline pipeline enables systematic evaluation of different TTS approaches:

### Currently Implemented
- **Orpheus TTS** - Advanced neural TTS with paralinguistic capabilities
- **Meditron Integration** - Medical knowledge + voice synthesis
- **Speech-to-Speech Interface** - Complete voice interaction pipeline with ASR
- **Evaluation Interface** - Web UI for testing and comparison

### Planned TTS Models for Testing
- **CoquiTTS** - Open-source TTS with custom voice training
- **Bark** - Transformer-based TTS with emotional control
- **XTTS** - Multilingual voice cloning

### Future: Conversational Speech Models
- **Qwen Fine-tuning** - Direct conversational speech generation
- **Context-Aware Prosody** - Tone adaptation for medical contexts
- **Dialogue State Integration** - Conversational history awareness

## Quick Start

### Prerequisites
- **Docker** & **Docker Compose** installed
- **16GB+ RAM** (for full Meditron-7B) or **8GB** (for DialoGPT baseline)
- **HuggingFace account** with access token

### Step 1: Configure Environment
Create a `.env` file in the project root:
```bash
HUGGINGFACE_TOKEN=your_token_here
MEDITRON_MODEL=microsoft/DialoGPT-medium  # or epfl-llm/meditron-7b
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft
WHISPER_MODEL=tiny  # or base, small, medium, large
```

### Step 2: Launch Services
```bash
# Make the startup script executable
chmod +x start-meditalk.sh

# Start all services
./start-meditalk.sh
```

The script will:
1. Check Docker installation
2. Build all service containers
3. Start the complete pipeline
4. Open the web interface at http://localhost:8080

### Step 3: Test the System
1. Navigate to **http://localhost:8080**
2. **Text Input**: Type a medical question directly
2-bis. **Voice Input**: Use speech recognition to ask questions by voice
   - Select between Browser API (recommended) or Whisper ASR (experimental)
   - Click the microphone button and speak your question
3. Select TTS voice (Tara/Nova)
4. Generate response and listen to audio
5. Evaluate quality and naturalness

## Deployment Options

### Local Development
Use Docker Compose for local testing and development:
```bash
./start-meditalk.sh
```
Access at http://localhost:8080

### EPFL RCP Cluster
Deploy MediTalk on the EPFL Research Computing Platform with GPU acceleration:

📚 **[Complete RCP Deployment Guide →](rcp/README-RCP.md)**

Quick start:
```bash
# Configure and build
cp rcp/configs/.env.rcp.example rcp/configs/.env.rcp
# Edit .env.rcp with your GASPAR username
./rcp/scripts/build_images.sh
./rcp/scripts/push_images.sh

# Deploy to cluster
./rcp/scripts/submit_all.sh

# Access web UI
./rcp/scripts/port_forward.sh
```

## Services Architecture

### Active Services (Phase 1)

| Service | Port | Purpose | Status |
|---------|------|---------|---------|
| **Web UI** | 8080 | Interactive evaluation interface | Active |
| **Meditron AI** | 5006 | Medical question answering | Active |
| **Orpheus TTS** | 5005 | Neural text-to-speech synthesis | Active |
| **Whisper ASR** | 5007 | Speech-to-text transcription | Active |

### Future Services (Phase 2+)

| Service | Port | Purpose | Status |
|---------|------|---------|---------|
| **CoquiTTS** | 5001 | Alternative TTS evaluation | Planned |
| **Bark TTS** | 5002 | Emotional TTS testing | Planned |
| **TTS Evaluator** | 5009 | Automated quality metrics | Planned |
| **Qwen Speech** | 5008 | Conversational speech generation | Research |

## Medical AI Configuration

### Phase 1: Lightweight Baseline
- **Model**: `microsoft/DialoGPT-medium`
- **Size**: ~345MB
- **RAM**: 1-2GB
- **Purpose**: Pipeline validation and basic TTS testing

### Phase 2: Full Medical Knowledge
- **Model**: `epfl-llm/meditron-7b`
- **Size**: ~14GB
- **RAM**: 14GB+
- **Purpose**: Baseline models evaluation with real Meditron model

### Phase 3: Conversational Speech
- **Model**: `Qwen-?` (fine-tuned)
- **Approach**: Direct conversational speech generation
- **Purpose**: Natural medical dialogue with context awareness using Meditron + fine-tuned Qwen

### Switching Models
Edit `.env` file:
```bash
# For lightweight testing
MEDITRON_MODEL=microsoft/DialoGPT-medium

# For full medical AI
MEDITRON_MODEL=epfl-llm/meditron-7b
```

Then restart services:
```bash
docker compose -f docker/docker-compose.yml restart meditron
```

## Research Methodology

### Phase 1: Baseline TTS Evaluation Pipeline (Current)
- [x] Set up Dockerized services
- [x] Set up Meditron + Orpheus TTS pipeline
- [x] Create web-based evaluation interface
- [x] Implement speech-to-speech pipeline with ASR
- [x] Add Browser Web Speech API integration
- [x] Add Whisper ASR service (experimental)
- [ ] Establish baseline performance metrics
- [ ] Document quality benchmarks

### Phase 2: Existing TTS/CSM Comparison and Evaluation (Next)
- [ ] Integrate CoquiTTS, Bark, XTTS
- [ ] Implement automated quality assessment
- [ ] Conduct comparative analysis (MOS)
- [ ] Document strengths/weaknesses of each approach

### Phase 3: Qwen Fine-tuning Research (Future)
- [ ] Collect (medical?) conversational speech data
- [ ] Design fine-tuning strategy for Qwen
- [ ] Implement Meditron + Qwen pipeline
- [ ] Evaluate conversational quality and medical tone
- [ ] Compare against baseline TTS approaches

## API Documentation

### Whisper ASR Service (Port 5007)

#### POST `/transcribe` - Speech-to-Text
```bash
curl -X POST "http://localhost:5007/transcribe" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "transcription": "What is hypertension?"
}
```

#### GET `/health` - Check Service Status
```bash
curl http://localhost:5007/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "tiny"
}
```

### Meditron Service (Port 5006)

#### POST `/ask` - Generate Medical Response
```bash
curl -X POST "http://localhost:5006/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is hypertension?",
    "max_length": 512,
    "temperature": 0.7,
    "generate_audio": true,
    "voice": "tara"
  }'
```

**Response:**
```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension, also known as high blood pressure...",
  "audio_file": "orpheus_output_1234.wav",
  "audio_url": "http://localhost:5005/audio/orpheus_output_1234.wav"
}
```

#### GET `/health` - Check Service Status
```bash
curl http://localhost:5006/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "microsoft/DialoGPT-medium"
}
```

### Orpheus TTS Service (Port 5005)

#### POST `/synthesize` - Generate Speech
```bash
curl -X POST "http://localhost:5005/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Medical response text here",
    "voice": "tara"
  }'
```

**Response:**
```json
{
  "audio_file": "orpheus_output_5678.wav",
  "audio_url": "http://localhost:5005/audio/orpheus_output_5678.wav"
}
```

#### GET `/audio/{filename}` - Download Audio File
```bash
curl -O http://localhost:5005/audio/orpheus_output_1234.wav
```

## Development

### Project Structure
```
MediTalk/
├── docker/
│   ├── docker-compose.yml      # Service orchestration
│   └── Dockerfile.base         # Base image configuration
├── services/
│   ├── modelMeditron/          # Medical AI service
│   │   ├── app.py             # FastAPI server
│   │   ├── meditron.py        # AI model wrapper
│   │   └── requirements.txt
│   ├── modelOrpheus/           # TTS service
│   │   ├── app.py             # FastAPI server
│   │   ├── infer.py           # TTS inference
│   │   └── requirements.txt
│   ├── modelWhisper/           # ASR service
│   │   ├── app.py             # FastAPI server
│   │   └── requirements.txt
│   └── webui/                  # Web interface
│       ├── app.py             # FastAPI backend
│       ├── templates/         # HTML templates
│       └── requirements.txt
├── outputs/
│   └── orpheus/               # Generated audio files
├── .env                       # Environment configuration
├── start-meditalk.sh         # Startup script
└── README.md                 # This file
```

### Service Management

#### Start/Stop Services
```bash
# Start all services
./start-meditalk.sh

# Stop all services
docker compose -f docker/docker-compose.yml down

# Restart specific service
docker compose -f docker/docker-compose.yml restart meditron
docker compose -f docker/docker-compose.yml restart orpheus
docker compose -f docker/docker-compose.yml restart webui
```

#### View Logs
```bash
# All services
docker compose -f docker/docker-compose.yml logs -f

# Specific service
docker compose -f docker/docker-compose.yml logs -f meditron
docker compose -f docker/docker-compose.yml logs -f orpheus
docker compose -f docker/docker-compose.yml logs -f webui
```

#### Health Monitoring
```bash
# Check service health
curl http://localhost:5006/health  # Meditron
curl http://localhost:5005/health  # Orpheus
curl http://localhost:5007/health  # Whisper ASR
curl http://localhost:8080         # Web UI
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | Yes | - | HuggingFace API authentication |
| `MEDITRON_MODEL` | No | `microsoft/DialoGPT-medium` | Medical AI model identifier |
| `ORPHEUS_MODEL` | No | `canopylabs/orpheus-3b-0.1-ft` | TTS model identifier |
| `WHISPER_MODEL` | No | `tiny` | Whisper ASR model size (tiny/base/small) |

### Hardware Requirements

#### Minimum (DialoGPT Basic Pipeline)
- **RAM**: 8GB
- **Storage**: 10GB
- **CPU**: 4 cores
- **GPU**: Not required

#### Recommended (Meditron-7B)
- **RAM**: 16GB+
- **Storage**: 50GB+
- **CPU**: 8+ cores
- **GPU**: Optional (CUDA-compatible for faster inference)

## Troubleshooting

### Common Issues

#### Memory Errors (Exit Code 137)
**Symptom**: Container exits with code 137
**Solution**: Switch to lighter model
```bash
echo "MEDITRON_MODEL=microsoft/DialoGPT-medium" >> .env
./start-meditalk.sh
```

#### HuggingFace Authentication Errors
**Symptom**: "401 Unauthorized" or token errors
**Solution**: Verify your token
```bash
curl -H "Authorization: Bearer $HUGGINGFACE_TOKEN" \
     https://huggingface.co/api/whoami-v2
```

#### Service Connection Issues
**Symptom**: Cannot connect to services
**Solution**: Check service status
```bash
docker compose -f docker/docker-compose.yml ps
curl http://localhost:5006/health
curl http://localhost:5005/health
```

#### Audio Generation Fails
**Symptom**: No audio output or synthesis errors
**Solution**: Check Orpheus logs
```bash
docker compose -f docker/docker-compose.yml logs orpheus

# Test direct synthesis
curl -X POST "http://localhost:5005/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "test audio", "voice": "tara"}'
```

#### Speech Recognition Issues
**Symptom**: Voice input not working
**Solution**: 
1. Use Browser Web Speech API (recommended) - works reliably in Chrome/Edge
2. For Whisper ASR issues, check logs:
```bash
docker compose -f docker/docker-compose.yml logs whisper

# Test Whisper transcription
curl -X POST "http://localhost:5007/transcribe" \
     -F "file=@test_audio.wav"
```

**Note**: Whisper ASR may have compatibility issues on some hardware. Browser Web Speech API is recommended as a temporary solution.

### Performance Optimization

#### Faster Response Times
- Reduce `max_length` parameter (shorter responses)
- Lower `temperature` value (more focused responses)
- Disable audio generation for text-only testing

#### Memory Management
- Use DialoGPT-medium for development
- Enable Docker resource limits in docker-compose.yml
- Monitor with `docker stats`

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

**MediTalk** - Empowering medical AI (Meditron) with natural conversational speech capabilities.

*Semester Research Project | Nicolas Teissier | EPFL*
