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
services=("webui" "modelMeditron" "modelWhisper" "modelOrpheus")

for service in "${services[@]}"; do
    stop_service "$service"
done

# Clean up any remaining uvicorn processes (safety measure)
pkill -f "uvicorn app:app" 2>/dev/null

echo ""
echo "All services stopped! ✓"
echo ""
