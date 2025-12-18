#!/bin/bash

echo "=================================="
echo "TTS Benchmark Runner"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "benchmark_tts.py" ]; then
    echo "❌ Error: Please run this script from the benchmark/tts directory"
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

# Run benchmark
echo ""
echo "Starting benchmark..."
echo "=================================="
echo ""

python3 benchmark_tts.py


if [ -f "benchmark.log" ]; then
    rm "benchmark.log"
fi

