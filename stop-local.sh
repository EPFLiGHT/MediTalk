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

# Stop services in reverse order
services=("webui" "modelMultiMeditron" "modelMeditron" "modelBark" "modelWhisper" "modelOrpheus")

for service in "${services[@]}"; do
    stop_service "$service"
done

# Clean up any remaining uvicorn processes (safety measure)
pkill -f "uvicorn app:app" 2>/dev/null

# Kill by process pattern first (most reliable)
echo ""
echo "Killing all service processes..."
pkill -9 -f "modelOrpheus/venv.*app.py" 2>/dev/null
pkill -9 -f "modelWhisper/venv.*app.py" 2>/dev/null
pkill -9 -f "modelBark/venv.*app.py" 2>/dev/null
pkill -9 -f "modelMeditron/venv.*app.py" 2>/dev/null
pkill -9 -f "modelMultiMeditron/venv.*app.py" 2>/dev/null
pkill -9 -f "webui/venv.*app.py" 2>/dev/null

# Wait a moment for processes to die
sleep 2

# Double-check ports and kill any remaining processes
echo "Checking for processes still blocking service ports..."
for port in 5005 5006 5007 5008 5009 8080; do
    # Try multiple methods to find and kill process on port
    # Method 1: Using ss
    pid=$(ss -tulpn 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ ! -z "$pid" ]; then
        echo "  Killing process $pid on port $port (found with ss)"
        kill -9 $pid 2>/dev/null
    fi
    
    # Method 2: Using fuser (if available)
    if command -v fuser &> /dev/null; then
        fuser -k -9 $port/tcp 2>/dev/null && echo "  Killed process on port $port (with fuser)"
    fi
    
    # Method 3: Grep through all python processes
    for pid in $(ps aux | grep "python.*app.py" | grep -v grep | awk '{print $2}'); do
        # Check if this process is listening on our port
        if netstat -tulpn 2>/dev/null | grep ":$port " | grep -q "$pid" || \
           ss -tulpn 2>/dev/null | grep ":$port " | grep -q "$pid"; then
            echo "  Killing process $pid on port $port (found with netstat)"
            kill -9 $pid 2>/dev/null
        fi
    done
done

echo ""
echo "All services stopped! ✓"
echo ""
