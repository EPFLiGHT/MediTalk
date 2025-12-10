#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env file with your tokens:"
    echo "  HF_DATA_TOKEN=hf_..."
    echo "  KAGGLE_API_TOKEN=KGAT_..."
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

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the data fetching script (it will load .env itself)
echo ""
echo "Running data fetching script..."
python data_fetching.py

echo ""
echo "Done!"
