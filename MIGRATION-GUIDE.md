# MediTalk Docker to Local Migration Guide

## Summary

Your MediTalk repository has been successfully configured to run **without Docker**! 🎉

## What Changed

### Files Created
1. **`setup-local.sh`** - Sets up virtual environments and installs dependencies
2. **`start-local.sh`** - Starts all services locally
3. **`stop-local.sh`** - Stops all running services
4. **`README-LOCAL.md`** - Comprehensive local deployment documentation

### Files Modified
1. **`services/webui/app.py`** - Updated to use localhost URLs instead of Docker container names
2. **`services/modelMeditron/app.py`** - Updated Orpheus service URL
3. **`services/modelOrpheus/app.py`** - Updated to handle local file paths for audio output

### Key Changes in Code

**Before (Docker):**
```python
# Services communicated using Docker container names
response = requests.post("http://orpheus:5005/synthesize", ...)
response = requests.post("http://meditron:5006/ask", ...)
```

**After (Local):**
```python
# Services use localhost with environment variable fallback
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
response = requests.post(f"{ORPHEUS_URL}/synthesize", ...)
```

**File Paths:**
- Docker: `/tmp/orpheus_audio/` (mounted volume)
- Local: `outputs/orpheus/` (local directory)

## Migration Steps

### For New Setup (First Time)

1. **Configure environment:**
   ```bash
   # Edit .env file with your HuggingFace token
   nano .env
   ```

2. **Run setup:**
   ```bash
   ./setup-local.sh
   ```

3. **Start services:**
   ```bash
   ./start-local.sh
   ```

4. **Access web UI:**
   ```
   http://localhost:8080
   ```

### If You Were Using Docker Before

1. **Stop Docker services:**
   ```bash
   docker compose -f docker/docker-compose.yml down
   ```

2. **Your existing `.env` file works as-is!**
   - No changes needed to environment variables

3. **Run local setup:**
   ```bash
   ./setup-local.sh
   ```

4. **Start local services:**
   ```bash
   ./start-local.sh
   ```

## Architecture Comparison

### Docker Deployment
```
┌─────────────────────────────────────────┐
│   Docker Compose                        │
│                                         │
│  ┌──────────┐  ┌──────────┐           │
│  │  webui   │  │ meditron │           │
│  │  :8080   │  │  :5006   │           │
│  └──────────┘  └──────────┘           │
│                                         │
│  ┌──────────┐  ┌──────────┐           │
│  │ orpheus  │  │ whisper  │           │
│  │  :5005   │  │  :5007   │           │
│  └──────────┘  └──────────┘           │
└─────────────────────────────────────────┘
```

### Local Deployment
```
┌─────────────────────────────────────────┐
│   Your macOS System                     │
│                                         │
│  services/webui/venv                    │
│    └─ uvicorn :8080                    │
│                                         │
│  services/modelMeditron/venv           │
│    └─ uvicorn :5006                    │
│                                         │
│  services/modelOrpheus/venv            │
│    └─ uvicorn :5005                    │
│                                         │
│  services/modelWhisper/venv            │
│    └─ uvicorn :5007                    │
└─────────────────────────────────────────┘
```

## Advantages of Local Deployment

### ✅ Pros
- **Faster startup** - No Docker container overhead
- **Easier debugging** - Direct access to Python processes
- **Better IDE integration** - Virtual envs work with VS Code/PyCharm
- **More flexible** - Easy to modify and test individual services
- **Lower memory usage** - No Docker daemon overhead
- **Simpler troubleshooting** - Standard Python stack traces

### ⚠️ Cons
- **Manual environment setup** - Need to manage Python versions
- **No automatic networking** - Services use localhost (already handled)
- **No resource limits** - Docker could limit memory/CPU per service
- **Platform-specific** - Requires compatible Python version on your OS

## Using Both Docker and Local

You can keep both deployment options! Just use different scripts:

**Docker:**
```bash
./start-meditalk.sh     # Original Docker script
```

**Local:**
```bash
./start-local.sh        # New local script
```

**Important:** Stop one before starting the other (port conflicts).

## Service Management

### Starting Services
```bash
./start-local.sh
```

### Stopping Services
```bash
./stop-local.sh
```

### Checking Logs
```bash
# Real-time logs
tail -f logs/webui.log
tail -f logs/modelMeditron.log
tail -f logs/modelOrpheus.log
tail -f logs/modelWhisper.log

# All logs
tail -f logs/*.log
```

### Health Checks
```bash
# Check if services are responding
curl http://localhost:8080/        # Web UI
curl http://localhost:5006/health  # Meditron
curl http://localhost:5005/health  # Orpheus
curl http://localhost:5007/health  # Whisper
```

### Process Management
```bash
# View running processes
ps aux | grep uvicorn

# Check ports
lsof -i :8080
lsof -i :5006
lsof -i :5005
lsof -i :5007
```

## Directory Structure

New directories created during local deployment:

```
MediTalk-NoDocker/
├── .pids/                     # Process IDs (auto-created)
│   ├── webui.pid
│   ├── modelMeditron.pid
│   ├── modelOrpheus.pid
│   └── modelWhisper.pid
│
├── logs/                      # Service logs (auto-created)
│   ├── webui.log
│   ├── modelMeditron.log
│   ├── modelOrpheus.log
│   └── modelWhisper.log
│
└── services/
    ├── webui/venv/           # Virtual environment
    ├── modelMeditron/venv/   # Virtual environment
    ├── modelOrpheus/venv/    # Virtual environment
    └── modelWhisper/venv/    # Virtual environment
```

## Troubleshooting Quick Reference

### "Port already in use"
```bash
# Kill processes on specific ports
lsof -ti:8080 | xargs kill -9
lsof -ti:5006 | xargs kill -9
lsof -ti:5005 | xargs kill -9
lsof -ti:5007 | xargs kill -9
```

### "Virtual environment not found"
```bash
./setup-local.sh
```

### "Module not found"
```bash
# Reinstall dependencies for a service
cd services/modelOrpheus
source venv/bin/activate
pip install -r requirements.txt
```

### "Out of memory"
```bash
# Use smaller models in .env
MEDITRON_MODEL=microsoft/DialoGPT-medium
WHISPER_MODEL=tiny
```

### "Permission denied"
```bash
chmod +x setup-local.sh start-local.sh stop-local.sh
```

## Next Steps

1. **Read the full documentation:** `README-LOCAL.md`
2. **Test the system:** Run `./start-local.sh` and access http://localhost:8080
3. **Monitor logs:** Use `tail -f logs/*.log` to watch service activity
4. **Optimize:** Adjust model sizes in `.env` based on your hardware

## Still Have Docker Files?

Yes! All Docker files remain unchanged:
- `docker/docker-compose.yml` - Still works
- `services/*/Dockerfile` - Still works
- `start-meditalk.sh` - Original Docker script still works

You can switch between Docker and local deployment anytime!

## Support

- **Local deployment docs:** `README-LOCAL.md`
- **General docs:** `README.md`
- **API docs:** `API_DOCUMENTATION.md`

## Summary

✅ **Your repo now supports BOTH Docker and local deployment!**

**To use local deployment:**
```bash
./setup-local.sh      # First time only
./start-local.sh      # Every time you want to start
./stop-local.sh       # When you want to stop
```

**To use Docker (still works):**
```bash
./start-meditalk.sh   # Original method
```

Happy coding! 🚀
