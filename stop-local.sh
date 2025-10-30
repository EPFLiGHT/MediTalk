#!/bin/bash

echo "Stopping MediTalk services..."
echo "=============================="

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

# Stop Orpheus monitor first if running
stop_service "orpheus-monitor"

# Stop services in reverse order
services=("webui-streamlit" "webui" "modelMultiMeditron" "modelMeditron" "modelBark" "modelCSM" "modelWhisper" "modelOrpheus")

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
pkill -9 -f "modelMeditron/venv" 2>/dev/null
pkill -9 -f "modelMultiMeditron/venv" 2>/dev/null
pkill -9 -f "webui/venv" 2>/dev/null

echo ""
echo "All services stopped! ✓"
echo ""
