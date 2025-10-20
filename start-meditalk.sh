#!/bin/bash

echo "Starting MediTalk - Medical AI with Voice"
echo "==========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "WARNING: Please edit .env file and add your HUGGINGFACE_TOKEN"
    echo "   Then run this script again."
    exit 1
fi

# Load environment variables
source .env

# Set defaults if not set
MEDITRON_MODEL=${MEDITRON_MODEL:-"microsoft/DialoGPT-medium"}

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_huggingface_token_here" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN not set in .env file!"
    echo "Please edit .env file and add your Hugging Face token:"
    echo "   1. Get token from: https://huggingface.co/settings/tokens"
    echo "   2. Request access to: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft"
    echo "   3. Add token to .env file"
    exit 1
fi

echo "Environment configured successfully"
echo "   - Meditron Model: $MEDITRON_MODEL"
echo "   - HF Token: ${HUGGINGFACE_TOKEN:0:10}..."

# Create necessary directories
mkdir -p outputs/orpheus

# Start the complete pipeline
echo "Starting services..."
# Export environment variables for docker-compose
export HUGGINGFACE_TOKEN
export MEDITRON_MODEL
# Also pass them directly to avoid warnings
HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" MEDITRON_MODEL="$MEDITRON_MODEL" docker compose -f docker/docker-compose.yml up -d

echo ""
echo "Services starting up..."
echo "   - Orpheus TTS: http://localhost:5005"
echo "   - Meditron AI: http://localhost:5006" 
echo "   - Web UI: http://localhost:8080"
echo ""
echo "Please wait for all services to initialize..."
echo "   This may take several minutes for AI models to load."
echo ""
echo "Check service status:"
echo "   docker compose -f docker/docker-compose.yml logs -f"
echo ""
echo "Once ready, open: http://localhost:8080"