#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
SERVICE_NAME="$1"

# Service definitions (service_name:friendly_name)
declare -A SERVICES=(
    ["orpheus-monitor"]="orpheus-monitor"
    ["webui-streamlit"]="webui-streamlit"
    ["webui"]="webui"
    ["modelMultiMeditron"]="modelMultiMeditron"
    ["modelBark"]="modelBark"
    ["modelCSM"]="modelCSM"
    ["modelWhisper"]="modelWhisper"
    ["modelOrpheus"]="modelOrpheus"
    ["modelQwen3Omni"]="modelQwen3Omni"
    ["modelNisqa"]="modelNisqa"
    ["controller"]="controller"
)

# Function to show usage
usage() {
    echo "Usage: $0 [service_name]"
    echo ""
    echo "Stop MediTalk services"
    echo ""
    echo "Arguments:"
    echo "  service_name    Optional. Name of specific service to stop."
    echo "                  If omitted, all services will be stopped."
    echo ""
    echo "Available services:"
    for service in "${!SERVICES[@]}"; do
        echo "  - $service"
    done | sort
    echo ""
    echo "Examples:"
    echo "  $0                      # Stop all services"
    echo "  $0 modelOrpheus         # Stop only Orpheus service"
    echo "  $0 webui-streamlit      # Stop only Streamlit UI"
    exit 0
}

# Show help if requested
if [ "$SERVICE_NAME" == "-h" ] || [ "$SERVICE_NAME" == "--help" ]; then
    usage
fi

# Function to stop a service
stop_service() {
    local service=$1
    local pid_file=".pids/$service.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $service (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 1
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force stopping $service..."
                kill -9 $pid 2>/dev/null
            fi
            
            echo "  ✓ $service stopped"
        else
            echo "$service is not running (stale PID file)"
        fi
        rm "$pid_file"
    else
        echo "$service: no PID file found"
    fi
}

# If specific service requested
if [ ! -z "$SERVICE_NAME" ]; then
    # Check if service exists
    if [ -z "${SERVICES[$SERVICE_NAME]}" ]; then
        echo "ERROR: Unknown service '$SERVICE_NAME'"
        echo ""
        echo "Available services:"
        for service in "${!SERVICES[@]}"; do
            echo "  - $service"
        done | sort
        exit 1
    fi
    
    echo "Stopping $SERVICE_NAME..."
    echo "=============================="
    stop_service "$SERVICE_NAME"
    echo ""
    echo "$SERVICE_NAME stopped! ✓"
    exit 0
fi

# Stop all services
echo "Stopping MediTalk services..."
echo "=============================="

# Stop Orpheus monitor first if running
stop_service "orpheus-monitor"

# Stop services in reverse order
services=("webui-streamlit" "webui" "modelQwen3Omni" "modelMultiMeditron" "modelBark" "modelCSM" "modelWhisper" "modelOrpheus" "controller")

for service in "${services[@]}"; do
    stop_service "$service"
done

# Clean up any remaining uvicorn and streamlit processes (safety measure)
pkill -f "uvicorn app:app" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null

# Kill any remaining Python processes that might be blocking ports
echo ""
echo "Checking for processes blocking service ports..."
for port in 5005 5006 5007 5008 5009 8080 8501; do
    # Find process using the port (works without lsof)
    pid=$(ss -tulpn 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ ! -z "$pid" ]; then
        echo "  Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null
    fi
done

# Also kill by process pattern as backup
pkill -9 -f "modelOrpheus/venv" 2>/dev/null
pkill -9 -f "modelWhisper/venv" 2>/dev/null
pkill -9 -f "modelBark/venv" 2>/dev/null
pkill -9 -f "modelMultiMeditron/venv" 2>/dev/null
pkill -9 -f "webui/venv" 2>/dev/null

echo ""
echo "All services stopped! ✓"
echo ""
