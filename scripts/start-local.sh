#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
SERVICE_NAME="$1"

# Service definitions with ports
declare -A SERVICE_PORTS=(
    ["modelOrpheus"]="5005"
    ["modelBark"]="5008"
    ["modelCSM"]="5010"
    ["modelWhisper"]="5007"
    ["modelMultiMeditron"]="5009"
    ["modelQwen3Omni"]="5014"
    ["webui-streamlit"]="8503"
    ["controller"]="8000"
)

# Function to show usage
usage() {
    echo "Usage: $0 [service_name]"
    echo ""
    echo "Start MediTalk services"
    echo ""
    echo "Arguments:"
    echo "  service_name    Optional. Name of specific service to start."
    echo "                  If omitted, all services will be started."
    echo ""
    echo "Available services:"
    for service in "${!SERVICE_PORTS[@]}"; do
        echo "  - $service (port ${SERVICE_PORTS[$service]})"
    done | sort
    echo ""
    echo "Examples:"
    echo "  $0                      # Start all services"
    echo "  $0 modelOrpheus         # Start only Orpheus service"
    echo "  $0 webui-streamlit      # Start only Streamlit UI"
    exit 0
}

# Show help if requested
if [ "$SERVICE_NAME" == "-h" ] || [ "$SERVICE_NAME" == "--help" ]; then
    usage
fi

echo "Starting MediTalk - Medical AI with Voice (Local Mode)"
echo "======================================================="
echo "Working directory: $PROJECT_ROOT"
echo ""

# Kill any orphaned processes from previous manual starts
echo "Cleaning up any orphaned processes..."
pkill -9 -f "services/controller.*python.*app.py" 2>/dev/null
pkill -9 -f "services/modelOrpheus.*python.*app.py" 2>/dev/null
pkill -9 -f "services/modelBark.*python.*app.py" 2>/dev/null
pkill -9 -f "services/modelWhisper.*python.*app.py" 2>/dev/null
pkill -9 -f "services/modelMultiMeditron.*python.*app.py" 2>/dev/null
pkill -9 -f "services/webui.*streamlit.*app.py" 2>/dev/null
sleep 2
echo "✓ Cleanup complete"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please run ./scripts/setup-local.sh first"
    exit 1
fi

# Load environment variables and automatically export them
set -a  # Auto-export all variables
source .env
set +a  # Stop auto-exporting

# Check if ffmpeg is available (required by Whisper)
# Check and install ffmpeg (required for Whisper audio processing)
echo ""
echo "Checking system dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ ffmpeg not found - required for Whisper audio transcription"
    echo "Installing ffmpeg..."
    
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y ffmpeg
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "ERROR: Could not install ffmpeg automatically."
        echo "Please install ffmpeg manually:"
        echo "  Ubuntu/Debian: sudo apt install ffmpeg"
        echo "  CentOS/RHEL: sudo yum install ffmpeg"
        echo "  macOS: brew install ffmpeg"
        exit 1
    fi
    
    if command -v ffmpeg &> /dev/null; then
        echo "✓ ffmpeg installed successfully"
    else
        echo "ERROR: ffmpeg installation failed"
        exit 1
    fi
else
    echo "✓ ffmpeg is installed ($(ffmpeg -version | head -n1))"
fi

# Ensure critical variables are set with defaults
export HUGGINGFACE_TOKEN
export MULTIMEDITRON_MODEL=${MULTIMEDITRON_MODEL:-"ClosedMeditron/Mulimeditron-End2End-CLIP-medical"}
export ORPHEUS_MODEL=${ORPHEUS_MODEL:-"canopylabs/orpheus-3b-0.1-ft"}
export WHISPER_MODEL=${WHISPER_MODEL:-"tiny"}

echo "Environment configured:"
echo "  - MultiMeditron Model: $MULTIMEDITRON_MODEL"
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
        echo "Please run ./scripts/setup-local.sh first"
        exit 1
    fi
    
    # Start the service in background
    # Restrict to GPUs 0-1-2 (available GPUs for services)
    cd "$service_dir"
    source venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0,1,2
    nohup uvicorn app:app --host 0.0.0.0 --port $port > "../../logs/$service.log" 2>&1 &
    echo $! > "../../$pid_file"
    unset CUDA_VISIBLE_DEVICES
    cd ../..
    
    echo "  ✓ $service started (PID: $(cat $pid_file))"
}

# Function to start a service with custom GPU selection
start_service_gpu() {
    local service=$1
    local port=$2
    local gpu=$3
    local service_dir="services/$service"
    local venv_dir="$service_dir/venv"
    local pid_file=".pids/$service.pid"
    
    echo "Starting $service on port $port (GPU $gpu)..."
    
    # Check if virtual environment exists
    if [ ! -d "$venv_dir" ]; then
        echo "ERROR: Virtual environment not found for $service"
        echo "Please run ./scripts/setup-local.sh first"
        exit 1
    fi
    
    # Start the service in background with CUDA_VISIBLE_DEVICES set
    cd "$service_dir"
    source venv/bin/activate
    export CUDA_VISIBLE_DEVICES=$gpu
    nohup uvicorn app:app --host 0.0.0.0 --port $port > "../../logs/$service.log" 2>&1 &
    echo $! > "../../$pid_file"
    unset CUDA_VISIBLE_DEVICES  # Clean up
    cd ../..
    
    echo "  ✓ $service started (PID: $(cat $pid_file)) on GPU $gpu"
}

# Function to start Streamlit service
start_streamlit() {
    local service=$1
    local port=$2
    local service_dir="services/$service"
    local venv_dir="$service_dir/venv"
    local pid_file=".pids/$service-streamlit.pid"
    
    echo "Starting $service Streamlit on port $port..."
    
    # Check if virtual environment exists
    if [ ! -d "$venv_dir" ]; then
        echo "ERROR: Virtual environment not found for $service"
        echo "Please run ./scripts/setup-local.sh first"
        exit 1
    fi
    
    # Start Streamlit in background (no GPU needed for web interface)
    cd "$service_dir"
    source venv/bin/activate
    nohup python -m streamlit run streamlit_app.py --server.port $port --server.address 0.0.0.0 --server.headless true > "../../logs/$service-streamlit.log" 2>&1 &
    echo $! > "../../$pid_file"
    cd ../..
    
    echo "  ✓ $service Streamlit started (PID: $(cat $pid_file))"
}

# Create logs directory
mkdir -p logs

# If specific service requested
if [ ! -z "$SERVICE_NAME" ]; then
    # Check if service exists
    if [ -z "${SERVICE_PORTS[$SERVICE_NAME]}" ]; then
        echo "ERROR: Unknown service '$SERVICE_NAME'"
        echo ""
        echo "Available services:"
        for service in "${!SERVICE_PORTS[@]}"; do
            echo "  - $service (port ${SERVICE_PORTS[$service]})"
        done | sort
        exit 1
    fi
    
    PORT="${SERVICE_PORTS[$SERVICE_NAME]}"
    
    echo "Starting $SERVICE_NAME on port $PORT..."
    echo ""
    
    # Handle special case for Streamlit
    if [ "$SERVICE_NAME" == "webui-streamlit" ]; then
        start_streamlit "webui" "$PORT"
    else
        start_service "$SERVICE_NAME" "$PORT"
    fi
    
    echo ""
    echo "=========================================="
    echo "$SERVICE_NAME started! ✓"
    echo ""
    echo "Service URL: http://localhost:$PORT"
    echo "Log file: logs/$SERVICE_NAME.log"
    echo ""
    echo "To check status:"
    echo "  tail -f logs/$SERVICE_NAME.log"
    echo ""
    echo "To stop service:"
    echo "  ./scripts/stop-local.sh $SERVICE_NAME"
    echo "=========================================="
    exit 0
fi

# Start all services in order (dependencies first)
echo ""
echo "Starting all services..."
echo ""

# 1. Start Controller (orchestration service - should start first)
echo "Starting Controller (orchestration service)..."
start_service "controller" 8000

sleep 2

# 2. Start Orpheus TTS
start_service "modelOrpheus" 5005

# 3. Start Bark TTS
start_service "modelBark" 5008

# 4. Start CSM TTS
start_service "modelCSM" 5010

# 5. Start Whisper ASR
start_service "modelWhisper" 5007

# Give TTS and ASR a moment to initialize
sleep 3

# 6. Start MultiMeditron
start_service "modelMultiMeditron" 5009

# Give AI models time to load
sleep 5

# 7. Start Streamlit UI (uses controller)
start_streamlit "webui" 8503

# 8. Start Qwen3-Omni model service
start_service "modelQwen3Omni" 5014

echo ""
echo "=========================================="
echo "All services started! "
echo ""
echo "Service URLs:"
echo "  - Streamlit Web Interface: http://localhost:8503"
echo "  - Controller (orchestration): http://localhost:8000"
echo "  - MultiMeditron AI (multimodal): http://localhost:5009"
echo "  - Orpheus TTS: http://localhost:5005"
echo "  - Bark TTS: http://localhost:5008"
echo "  - CSM TTS: http://localhost:5010"
echo "  - Whisper ASR: http://localhost:5007"
echo "  - Qwen3-Omni AI: http://localhost:5014"
echo ""
echo "Logs are available in the logs/ directory"
echo ""
echo "To check service status:"
echo "  tail -f logs/controller.log"
echo "  tail -f logs/webui-streamlit.log"
echo "  tail -f logs/modelMultiMeditron.log"
echo "  tail -f logs/modelOrpheus.log"
echo "  tail -f logs/modelBark.log"
echo "  tail -f logs/modelCSM.log"
echo "  tail -f logs/modelWhisper.log"
echo "  tail -f logs/modelQwen3Omni.log"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop-local.sh"
echo "=========================================="
echo ""
echo "Waiting for services to be ready..."
sleep 5
echo ""
echo "Opening Streamlit web interface..."
open http://localhost:8503 2>/dev/null || xdg-open http://localhost:8503 2>/dev/null || echo "Please open http://localhost:8503 in your browser"
echo ""

# Start Orpheus monitoring in background
echo "Starting Orpheus connection monitor..."
(
    while true; do
        sleep 30  # Check every 30 seconds
        
        # Check if Orpheus log has connection timeout errors in the last 2 minutes
        if tail -n 100 logs/modelOrpheus.log 2>/dev/null | grep -q "Connection timed out"; then
            echo "[$(date)] Orpheus connection timeout detected. Restarting Orpheus..." >> logs/orpheus-monitor.log
            
            # Restart Orpheus
            if [ -f .pids/modelOrpheus.pid ]; then
                kill $(cat .pids/modelOrpheus.pid) 2>/dev/null
                sleep 2
                kill -9 $(cat .pids/modelOrpheus.pid) 2>/dev/null
            fi
            
            # Start Orpheus with environment variables
            cd services/modelOrpheus
            source venv/bin/activate
            export $(grep -v '^#' ../../.env | xargs)
            nohup uvicorn app:app --host 0.0.0.0 --port 5005 > ../../logs/modelOrpheus.log 2>&1 &
            echo $! > ../../.pids/modelOrpheus.pid
            cd ../..
            
            echo "[$(date)] Orpheus restarted with PID $(cat .pids/modelOrpheus.pid)" >> logs/orpheus-monitor.log
            
            # Wait before checking again to avoid rapid restarts
            sleep 120
        fi
    done
) &

MONITOR_PID=$!
echo $MONITOR_PID > .pids/orpheus-monitor.pid
echo "✓ Orpheus monitor started (PID: $MONITOR_PID)"
echo "  Monitor logs: logs/orpheus-monitor.log"
