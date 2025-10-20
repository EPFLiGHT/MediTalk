# MediTalk - Local Deployment (No Docker)

This guide explains how to run MediTalk without Docker using Python virtual environments.

## Overview

MediTalk consists of 4 microservices that communicate via HTTP:
- **Web UI** (port 8080) - User interface
- **Meditron** (port 5006) - Medical AI language model
- **Orpheus** (port 5005) - Text-to-Speech service
- **Whisper** (port 5007) - Speech-to-Text service

## Prerequisites

### System Requirements
- **macOS** (Linux and Windows WSL also supported)
- **Python 3.10+** installed
- **16GB+ RAM** (for Meditron-7B) or **8GB** (for DialoGPT baseline)
- **20GB+ disk space** (for models)
- **HuggingFace account** with access token

### Install Python Dependencies
Ensure you have Python 3.10 or higher:
```bash
python3 --version
```

## Quick Start

### Step 1: Configure Environment

Create a `.env` file in the project root:
```bash
# Copy the example if it exists
cp .env.example .env

# Or create manually
cat > .env << EOF
HUGGINGFACE_TOKEN=your_token_here
MEDITRON_MODEL=microsoft/DialoGPT-medium
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft
WHISPER_MODEL=tiny
EOF
```

**Important:** Get your HuggingFace token:
1. Create account at https://huggingface.co
2. Generate token at https://huggingface.co/settings/tokens
3. Request access to https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
4. Add token to `.env` file

### Step 2: Setup Environment

Run the setup script to create virtual environments and install dependencies:
```bash
chmod +x setup-local.sh
./setup-local.sh
```

This will:
- Create a virtual environment for each service
- Install all Python dependencies
- Verify your configuration

**Note:** This may take 10-20 minutes depending on your internet speed and CPU.

### Step 3: Start MediTalk

```bash
chmod +x start-local.sh
./start-local.sh
```

The script will:
1. Start all services in the background
2. Create log files in `logs/` directory
3. Open the web interface at http://localhost:8080

### Step 4: Use MediTalk

Navigate to http://localhost:8080 in your browser to:
- Ask medical questions via text
- Use voice input (Web Speech API or Whisper)
- Listen to AI-generated voice responses
- Test different TTS voices

## Managing Services

### Check Service Status

View logs in real-time:
```bash
# All services
tail -f logs/*.log

# Individual services
tail -f logs/webui.log
tail -f logs/modelMeditron.log
tail -f logs/modelOrpheus.log
tail -f logs/modelWhisper.log
```

### Check if Services are Running

```bash
# Check processes
ps aux | grep uvicorn

# Check ports
lsof -i :8080  # Web UI
lsof -i :5006  # Meditron
lsof -i :5005  # Orpheus
lsof -i :5007  # Whisper
```

### Test Individual Services

```bash
# Health checks
curl http://localhost:5005/health  # Orpheus TTS
curl http://localhost:5006/health  # Meditron
curl http://localhost:5007/health  # Whisper ASR
```

### Stop Services

```bash
chmod +x stop-local.sh
./stop-local.sh
```

This will gracefully shut down all services.

## Troubleshooting

### Services Won't Start

**Check Python version:**
```bash
python3 --version  # Should be 3.10 or higher
```

**Check virtual environments:**
```bash
ls services/*/venv  # Should show venv directories
```

**Reinstall dependencies:**
```bash
./setup-local.sh
```

### Port Already in Use

If you see "Address already in use" errors:
```bash
# Find and kill processes on specific ports
lsof -ti:8080 | xargs kill -9
lsof -ti:5006 | xargs kill -9
lsof -ti:5005 | xargs kill -9
lsof -ti:5007 | xargs kill -9
```

### Out of Memory

**For Meditron-7B:**
- Requires 16GB+ RAM
- Consider using smaller model: `MEDITRON_MODEL=microsoft/DialoGPT-medium`

**For DialoGPT:**
- Requires 8GB RAM
- Already set as default in `.env`

### Model Loading Issues

**HuggingFace authentication:**
```bash
# Verify token is set
echo $HUGGINGFACE_TOKEN

# Test authentication
pip install huggingface_hub
python3 -c "from huggingface_hub import login; login('your_token_here')"
```

**Model access:**
- Ensure you've requested access to Orpheus model
- Check https://huggingface.co/canopylabs/orpheus-3b-0.1-ft

### Audio Files Not Found

Check output directory:
```bash
ls -la outputs/orpheus/
```

Ensure directory exists:
```bash
mkdir -p outputs/orpheus
chmod 755 outputs/orpheus
```

### Whisper Errors

If Whisper crashes or gives segfaults:
```bash
# Try smaller model
export WHISPER_MODEL=tiny

# Or disable Whisper and use browser speech recognition
```

## Performance Optimization

### Use Smaller Models

Edit `.env`:
```bash
# Faster startup, less memory
MEDITRON_MODEL=microsoft/DialoGPT-medium
WHISPER_MODEL=tiny

# Better quality, more resources
MEDITRON_MODEL=epfl-llm/meditron-7b
WHISPER_MODEL=base
```

### GPU Acceleration

If you have a CUDA-compatible GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Reduce Memory Usage

Set environment variables before starting:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Architecture Comparison

### Docker vs Local Deployment

| Feature | Docker | Local |
|---------|--------|-------|
| Setup | `docker compose up` | `./setup-local.sh && ./start-local.sh` |
| Isolation | Containers | Virtual environments |
| Networking | Container names | localhost URLs |
| Logs | `docker logs` | `logs/*.log` files |
| Resources | Docker overhead | Direct system access |
| Debugging | Container shell | Direct file access |

### Service Communication

**Docker mode:**
```
webui → http://meditron:5006
webui → http://orpheus:5005
meditron → http://orpheus:5005
```

**Local mode:**
```
webui → http://localhost:5006
webui → http://localhost:5005
meditron → http://localhost:5005
```

## Development

### Running Individual Services

For development, you can run services individually:

```bash
# Orpheus TTS
cd services/modelOrpheus
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 5005 --reload

# Meditron
cd services/modelMeditron
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 5006 --reload

# Whisper
cd services/modelWhisper
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 5007 --reload

# Web UI
cd services/webui
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The `--reload` flag enables auto-reload on code changes.

### Adding New Dependencies

```bash
# Add to requirements.txt
cd services/modelOrpheus
source venv/bin/activate
echo "new-package==1.0.0" >> requirements.txt
pip install -r requirements.txt
```

## File Structure

```
MediTalk-NoDocker/
├── .env                    # Environment configuration
├── setup-local.sh          # Setup script
├── start-local.sh          # Start script
├── stop-local.sh           # Stop script
├── logs/                   # Service logs
│   ├── webui.log
│   ├── modelMeditron.log
│   ├── modelOrpheus.log
│   └── modelWhisper.log
├── .pids/                  # Process IDs
│   ├── webui.pid
│   ├── modelMeditron.pid
│   ├── modelOrpheus.pid
│   └── modelWhisper.pid
├── outputs/
│   └── orpheus/           # Generated audio files
├── models/                # Downloaded models cache
└── services/
    ├── webui/
    │   ├── venv/          # Virtual environment
    │   ├── app.py
    │   └── requirements.txt
    ├── modelMeditron/
    │   ├── venv/
    │   ├── app.py
    │   └── requirements.txt
    ├── modelOrpheus/
    │   ├── venv/
    │   ├── app.py
    │   └── requirements.txt
    └── modelWhisper/
        ├── venv/
        ├── app.py
        └── requirements.txt
```

## FAQ

**Q: Can I use both Docker and local deployment?**  
A: Yes! The code supports both. Just make sure to stop one before starting the other to avoid port conflicts.

**Q: Which is better - Docker or local?**  
A: Docker is easier for production/deployment. Local is better for development and debugging.

**Q: How do I switch back to Docker?**  
A: Use the original scripts: `./start-meditalk.sh` (requires Docker installed)

**Q: Can I run on Linux or Windows?**  
A: Yes! The scripts work on Linux and Windows WSL. For Windows, use WSL2 or adapt scripts for PowerShell.

**Q: How much disk space do I need?**  
A: ~20GB for models and dependencies. More if using larger models.

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review this README and troubleshooting section
3. Check the main README.md for general MediTalk documentation
4. Open an issue on GitHub

## License

Same as main MediTalk project.
