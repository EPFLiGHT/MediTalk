#!/bin/bash

echo "=================================================="
echo "  Whisper ASR Benchmarking Script"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "benchmark_whisper.py" ]; then
    echo "❌ Error: Please run this script from the benchmark/whisper directory"
    exit 1
fi

# Check if config file is provided
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: Config file 'config.yaml' not found in the current directory"
    echo "Please provide a valid config file (cf README.md)."
    echo "Tip: You can copy from 'config_example.yaml'."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "Dependencies installed"
echo ""

# Check if Whisper service is running
echo "Checking Whisper service health..."
HEALTH_CHECK=$(curl -s http://localhost:5007/health 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "Whisper service is running"
    echo "   Status: $HEALTH_CHECK"
    echo ""
    
    # Run benchmark
    echo "Starting benchmark..."
    echo "=================================="
    echo ""

    python benchmark_whisper.py

    echo ""
    echo "=================================="
    echo "Benchmark complete!"
    echo "Results saved to: results/"
    echo "=================================="
    
else
    echo "❌ Whisper service is not running"
    echo ""
    echo "Please start the Whisper service first:"
    echo ""
    echo "  cd ../../scripts"
    echo "  ./start-local.sh modelWhisper"
    echo ""
    exit 1
fi
