#!/bin/bash

echo "Starting MediTalk - Medical AI with Voice (Local Mode)"
echo "======================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please run ./setup-local.sh first"
    exit 1
fi

# Load environment variables
source .env
export HUGGINGFACE_TOKEN
export MEDITRON_MODEL=${MEDITRON_MODEL:-"microsoft/DialoGPT-medium"}
export ORPHEUS_MODEL=${ORPHEUS_MODEL:-"canopylabs/orpheus-3b-0.1-ft"}
export WHISPER_MODEL=${WHISPER_MODEL:-"tiny"}

echo "Environment configured:"
echo "  - Meditron Model: $MEDITRON_MODEL"
echo "  - Orpheus Model: $ORPHEUS_MODEL"
echo "  - Whisper Model: $WHISPER_MODEL"
echo ""

# Create PID directory
mkdir -p .pids

# Function to start a service
start_service() {
    local service=$1
    local port=$2
    local service_dir="services/$service"
    local venv_dir="$service_dir/venv"
    local pid_file=".pids/$service.pid"
    
    echo "Starting $service on port $port..."
    
    # Check if virtual environment exists
    if [ ! -d "$venv_dir" ]; then
        echo "ERROR: Virtual environment not found for $service"
        echo "Please run ./setup-local.sh first"
        exit 1
    fi
    
    # Start the service in background
    cd "$service_dir"
    source venv/bin/activate
    nohup uvicorn app:app --host 0.0.0.0 --port $port > "../../logs/$service.log" 2>&1 &
    echo $! > "../../$pid_file"
    cd ../..
    
    echo "  ✓ $service started (PID: $(cat $pid_file))"
}

# Create logs directory
mkdir -p logs

# Start services in order (dependencies first)
echo ""
echo "Starting services..."
echo ""

# 1. Start Orpheus TTS (required by Meditron)
start_service "modelOrpheus" 5005

# 2. Start Whisper ASR
start_service "modelWhisper" 5007

# Give TTS and ASR a moment to initialize
sleep 3

# 3. Start Meditron (depends on Orpheus)
start_service "modelMeditron" 5006

# Give Meditron time to load model
sleep 5

# 4. Start WebUI (depends on all services)
start_service "webui" 8080

echo ""
echo "=========================================="
echo "All services started! "
echo ""
echo "Service URLs:"
echo "  - Web Interface: http://localhost:8080"
echo "  - Meditron AI: http://localhost:5006"
echo "  - Orpheus TTS: http://localhost:5005"
echo "  - Whisper ASR: http://localhost:5007"
echo ""
echo "Logs are available in the logs/ directory"
echo ""
echo "To check service status:"
echo "  tail -f logs/webui.log"
echo "  tail -f logs/modelMeditron.log"
echo "  tail -f logs/modelOrpheus.log"
echo "  tail -f logs/modelWhisper.log"
echo ""
echo "To stop all services:"
echo "  ./stop-local.sh"
echo "=========================================="
echo ""
echo "Waiting for services to be ready..."
sleep 5
echo ""
echo "Opening web interface..."
open http://localhost:8080 2>/dev/null || xdg-open http://localhost:8080 2>/dev/null || echo "Please open http://localhost:8080 in your browser"
