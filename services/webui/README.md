# WebUI Service

Basic web interface for the MediTalk medical conversation AI project using Streamlit.

## Setup

```bash
cd services/webui

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the interface on port 8503:

```bash
streamlit run streamlit_app.py
```

Access the interface at http://localhost:8503

## Features

- Interactive medical conversation interface
- Audio recording and playback
- Real-time transcription and response generation
- Conversation history management
- Service health monitoring

Requires the controller service to be running on port 8000.
