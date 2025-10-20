# Docker vs Local Deployment - Side by Side Comparison

## Starting Services

### Docker Method
```bash
# Check Docker is running
docker --version

# Start services
./start-meditalk.sh

# Behind the scenes:
# - Reads docker/docker-compose.yml
# - Builds/pulls Docker images
# - Creates Docker network
# - Starts 4 containers
# - Mounts volumes for data
```

### Local Method
```bash
# Check Python is installed
python3 --version

# Setup (first time only)
./setup-local.sh

# Start services
./start-local.sh

# Behind the scenes:
# - Activates virtual environments
# - Starts 4 uvicorn processes
# - Logs to logs/ directory
# - Uses local outputs/ directory
```

---

## Service Communication

### Docker
```yaml
services:
  webui:
    depends_on:
      - meditron
      - orpheus
    # Uses Docker DNS
    # http://meditron:5006
    # http://orpheus:5005
  
  meditron:
    depends_on:
      - orpheus
    # Uses Docker DNS
    # http://orpheus:5005
```

**Code:**
```python
# Docker networking
response = requests.post(
    "http://orpheus:5005/synthesize",
    json={"text": answer}
)
```

### Local
```bash
# All services on localhost
webui     → localhost:8080
meditron  → localhost:5006
orpheus   → localhost:5005
whisper   → localhost:5007
```

**Code:**
```python
# Environment variable with fallback
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")

response = requests.post(
    f"{ORPHEUS_URL}/synthesize",
    json={"text": answer}
)
```

---

## File Structure

### Docker
```
MediTalk-NoDocker/
├── docker/
│   ├── docker-compose.yml    ← Service definitions
│   └── Dockerfile.base
├── services/
│   ├── webui/
│   │   ├── Dockerfile        ← Build instructions
│   │   └── requirements.txt
│   ├── modelMeditron/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── ...
├── outputs/
│   └── orpheus/              ← Mounted from /tmp in container
└── models/                   ← Mounted to /models in containers
```

### Local
```
MediTalk-NoDocker/
├── services/
│   ├── webui/
│   │   ├── venv/             ← Virtual environment (NEW)
│   │   ├── app.py            ← Modified for localhost
│   │   └── requirements.txt
│   ├── modelMeditron/
│   │   ├── venv/             ← Virtual environment (NEW)
│   │   ├── app.py            ← Modified for localhost
│   │   └── requirements.txt
│   └── ...
├── logs/                     ← Service logs (NEW)
│   ├── webui.log
│   ├── modelMeditron.log
│   └── ...
├── .pids/                    ← Process IDs (NEW)
│   ├── webui.pid
│   └── ...
├── outputs/
│   └── orpheus/              ← Direct local directory
└── models/                   ← Direct local directory
```

---

## Managing Services

### Docker

**Start:**
```bash
./start-meditalk.sh
# or
docker compose -f docker/docker-compose.yml up -d
```

**Stop:**
```bash
docker compose -f docker/docker-compose.yml down
```

**View logs:**
```bash
docker compose -f docker/docker-compose.yml logs -f
docker logs meditalk-webui
```

**Check status:**
```bash
docker ps
docker compose -f docker/docker-compose.yml ps
```

**Restart one service:**
```bash
docker compose -f docker/docker-compose.yml restart meditron
```

### Local

**Start:**
```bash
./start-local.sh
```

**Stop:**
```bash
./stop-local.sh
```

**View logs:**
```bash
tail -f logs/webui.log
tail -f logs/*.log  # All logs
```

**Check status:**
```bash
./health-check.sh
# or
ps aux | grep uvicorn
lsof -i :8080,5005,5006,5007
```

**Restart one service:**
```bash
# Stop
kill $(cat .pids/meditron.pid)

# Start
cd services/modelMeditron
source venv/bin/activate
nohup uvicorn app:app --host 0.0.0.0 --port 5006 > ../../logs/modelMeditron.log 2>&1 &
```

---

## Development Workflow

### Docker

**Edit code:**
1. Modify `services/webui/app.py`
2. Rebuild container: `docker compose build webui`
3. Restart: `docker compose up -d webui`

**Add dependency:**
1. Add to `requirements.txt`
2. Rebuild: `docker compose build webui`
3. Restart: `docker compose up -d webui`

**Debug:**
```bash
docker exec -it meditalk-webui bash
# Now inside container
python3
>>> import requests
>>> requests.get("http://orpheus:5005/health")
```

### Local

**Edit code:**
1. Modify `services/webui/app.py`
2. Restart: `./stop-local.sh && ./start-local.sh`
   *or use `--reload` flag during development*

**Add dependency:**
1. Add to `requirements.txt`
2. Install: `cd services/webui && source venv/bin/activate && pip install -r requirements.txt`
3. Restart service

**Debug:**
```bash
cd services/webui
source venv/bin/activate
python3
>>> import requests
>>> requests.get("http://localhost:5005/health")
```

**Development mode (auto-reload):**
```bash
cd services/webui
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
# Now changes auto-reload!
```

---

## Resource Usage

### Docker

**Memory:**
- Docker daemon: ~500MB
- Each container: 2-8GB (depending on model)
- Total: ~10-20GB

**Disk:**
- Docker images: ~5-10GB
- Model cache: ~10-15GB
- Total: ~15-25GB

**Isolation:**
- ✅ Each service in separate container
- ✅ Resource limits enforced
- ✅ Network isolation
- ❌ Some overhead from Docker

### Local

**Memory:**
- Python processes: 2-8GB each
- No Docker overhead
- Total: ~8-16GB

**Disk:**
- Virtual environments: ~1-2GB
- Model cache: ~10-15GB
- Total: ~11-17GB

**Isolation:**
- ✅ Each service in separate venv
- ❌ No enforced resource limits
- ❌ Shared network namespace
- ✅ No Docker overhead

---

## Compatibility

### Docker

**Pros:**
- ✅ Same environment on any OS
- ✅ Easy deployment to servers
- ✅ Built-in service orchestration
- ✅ Resource limits built-in
- ✅ Production-ready

**Cons:**
- ❌ Requires Docker installed
- ❌ Additional resource overhead
- ❌ Slower iteration during development
- ❌ More complex debugging

### Local

**Pros:**
- ✅ No Docker required
- ✅ Faster development iteration
- ✅ Direct debugging access
- ✅ Better IDE integration
- ✅ Lower resource usage

**Cons:**
- ❌ Platform-dependent (Python versions)
- ❌ Manual environment management
- ❌ No built-in resource limits
- ❌ Requires Python 3.10+ installed

---

## When to Use Which?

### Use Docker When:
- 🚀 Deploying to production
- 🔄 Need consistent environments across machines
- 👥 Multiple developers with different OS
- 🛡️ Need isolation and resource limits
- 📦 Don't want to manage Python versions

### Use Local When:
- 💻 Active development on your Mac
- 🐛 Debugging complex issues
- ⚡ Want faster iteration cycles
- 🎯 Testing individual services
- 📊 Profiling performance

### Use Both When:
- 🔬 Develop locally, deploy with Docker
- ✅ Test changes locally before containerizing
- 🔄 Switch based on task requirements

---

## Summary

| Feature | Docker | Local |
|---------|--------|-------|
| Setup time | 5-10 min | 10-20 min |
| Start time | 2-3 min | 30-60 sec |
| Memory usage | Higher | Lower |
| Debugging | Container shell | Direct access |
| Hot reload | Rebuild required | Native support |
| Production-ready | ✅ Yes | ⚠️ Requires setup |
| Development-friendly | ⚠️ Slower | ✅ Faster |

**The Good News:** Your repo now supports BOTH! 🎉

Choose the right tool for the right job, or use both in combination!
