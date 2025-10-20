#!/bin/bash

echo "Setting up MediTalk for local (non-Docker) deployment"
echo "======================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your configuration"
    exit 1
fi

# Load environment variables
source .env

# Check HUGGINGFACE_TOKEN
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_huggingface_token_here" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN not set in .env file!"
    echo "Please edit .env file and add your Hugging Face token"
    exit 1
fi

echo "✓ Environment configured successfully"

# Create necessary directories
mkdir -p outputs/orpheus
mkdir -p models
echo "✓ Created output directories"

# Setup virtual environments for each service
services=("webui" "modelMeditron" "modelOrpheus" "modelWhisper")

for service in "${services[@]}"; do
    echo ""
    echo "Setting up $service..."
    
    service_dir="services/$service"
    venv_dir="$service_dir/venv"
    
    # Create virtual environment
    if [ ! -d "$venv_dir" ]; then
        echo "  Creating virtual environment..."
        python3 -m venv "$venv_dir"
    else
        echo "  Virtual environment already exists"
    fi
    
    # Activate and install requirements
    echo "  Installing dependencies..."
    source "$venv_dir/bin/activate"
    pip install --upgrade pip > /dev/null 2>&1
    
    if [ -f "$service_dir/requirements.txt" ]; then
        pip install -r "$service_dir/requirements.txt"
    else
        echo "  WARNING: No requirements.txt found for $service"
    fi
    
    deactivate
    echo "  ✓ $service ready"
done

echo ""
echo "=========================================="
echo "Setup complete! 🎉"
echo ""
echo "To start MediTalk:"
echo "  ./start-local.sh"
echo ""
echo "To stop MediTalk:"
echo "  ./stop-local.sh"
echo "=========================================="
