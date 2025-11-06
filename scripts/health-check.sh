#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  MediTalk Health Check"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    
    echo -n "Checking $name... "
    
    if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        
        # Try to get health status if available
        if [[ "$url" == *"/health" ]]; then
            status=$(curl -s "$url" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            if [ -n "$status" ]; then
                echo "  Status: $status"
            fi
        fi
    else
        echo -e "${RED}✗ Not responding${NC}"
    fi
}

# Check if services are running
echo "Service Status:"
echo "----------------------------------------"
check_service "Web UI       " "http://localhost:8080/"
check_service "Meditron AI  " "http://localhost:5006/health"
check_service "Orpheus TTS  " "http://localhost:5005/health"
check_service "Whisper ASR  " "http://localhost:5007/health"

echo ""
echo "----------------------------------------"
echo "Process Information:"
echo "----------------------------------------"

# Check for running processes
if pgrep -f "uvicorn app:app" > /dev/null; then
    echo -e "${GREEN}Found running uvicorn processes:${NC}"
    ps aux | grep "[u]vicorn app:app" | awk '{printf "  PID: %-6s Port: %s\n", $2, $0}' | grep -o "PID.*[0-9]\{4\}"
else
    echo -e "${YELLOW}No uvicorn processes found${NC}"
    echo "Services may not be running. Try: ./scripts/start-local.sh"
fi

echo ""
echo "----------------------------------------"
echo "Port Usage:"
echo "----------------------------------------"

for port in 8080 5006 5005 5007; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo -e "${GREEN}Port $port: In use${NC}"
    else
        echo -e "${YELLOW}Port $port: Available${NC}"
    fi
done

echo ""
echo "----------------------------------------"
echo "Log Files:"
echo "----------------------------------------"

if [ -d "logs" ]; then
    for log in logs/*.log; do
        if [ -f "$log" ]; then
            size=$(wc -c < "$log" | xargs)
            lines=$(wc -l < "$log" | xargs)
            echo "  $(basename $log): $lines lines ($size bytes)"
        fi
    done
else
    echo -e "${YELLOW}No logs directory found${NC}"
fi

echo ""
echo "=========================================="

# Quick recommendations
if ! pgrep -f "uvicorn app:app" > /dev/null; then
    echo ""
    echo "Tip: Services are not running. Start them with:"
    echo "   ./scripts/start-local.sh"
fi

if [ -f ".env" ]; then
    echo ""
    echo "✓ Configuration file (.env) found"
else
    echo ""
    echo "/!\  Warning: No .env file found!"
    echo "   Create one from .env.example and add your HUGGINGFACE_TOKEN"
fi

echo ""
