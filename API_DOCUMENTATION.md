# MediTalk API Documentation

## 🏥 Complete Medical AI Pipeline

MediTalk combines Meditron-7B (medical LLM) with Orpheus TTS to create a voice-enabled medical AI assistant.

### Architecture

```
User Question → Meditron-7B → Medical Response → Orpheus TTS → Audio Response
```

## 🚀 Quick Start

1. **Start all services:**
   ```bash
   ./start-meditalk.sh
   ```

2. **Open Web UI:** http://localhost:8080

3. **Or use API directly:** http://localhost:5006

## 📡 API Endpoints

### Meditron Service (Port 5006)

#### POST /ask
Ask a medical question with optional audio generation.

**Request:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "max_length": 512,
  "temperature": 0.7,
  "generate_audio": true,
  "voice": "tara"
}
```

**Response:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "answer": "Diabetes symptoms include increased thirst, frequent urination...",
  "audio_file": "orpheus_output_1234.wav",
  "audio_url": "http://localhost:5005/audio/orpheus_output_1234.wav"
}
```

#### GET /ask
Simple query interface:
```
GET /ask?question=What%20is%20diabetes&generate_audio=true
```

#### GET /health
Check service status:
```json
{"status": "healthy", "model": "meditron-7b"}
```

### Orpheus TTS Service (Port 5005)

#### POST /synthesize
Generate speech from text:
```json
{
  "text": "Medical response text",
  "voice": "tara"
}
```

#### GET /audio/{filename}
Download generated audio files:
```
GET /audio/orpheus_output_1234.wav
```

### Web UI (Port 8080)

#### GET /
Main web interface for interacting with the medical AI.

## 🧪 Testing Examples

### 1. Simple Medical Question
```bash
curl -X POST "http://localhost:5006/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is hypertension?",
    "generate_audio": true
  }'
```

### 2. Complex Medical Query
```bash
curl -X POST "http://localhost:5006/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the difference between Type 1 and Type 2 diabetes, including symptoms and treatment approaches.",
    "max_length": 800,
    "temperature": 0.5,
    "generate_audio": true,
    "voice": "tara"
  }'
```

### 3. Text-Only Response
```bash
curl "http://localhost:5006/ask?question=What%20causes%20headaches&generate_audio=false"
```

## 🔧 Configuration

### Environment Variables
```bash
HUGGINGFACE_TOKEN=your_token_here  # Required for both Meditron and Orpheus
```

### Service Ports
- **Web UI**: 8080
- **Meditron AI**: 5006  
- **Orpheus TTS**: 5005

### File Outputs
Generated audio files are saved to:
- **Container**: `/tmp/orpheus_audio/`
- **Host**: `outputs/orpheus/`

## 🏥 Medical AI Features

### Meditron-7B Capabilities
- Medical question answering
- Symptom analysis
- Treatment recommendations
- Drug information
- Medical education content

### Voice Synthesis
- Natural speech generation
- Medical terminology pronunciation
- Configurable voice parameters

## 🛠️ Service Management

### Check Service Health
```bash
# All services
docker-compose ps

# Individual health checks
curl http://localhost:5006/health  # Meditron
curl http://localhost:5005/health  # Orpheus
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f meditron
docker-compose logs -f orpheus
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart meditron
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check HUGGINGFACE_TOKEN is set
   - Verify model access permissions on Hugging Face
   - Check available memory/disk space (cf. requirements in README)

2. **Audio Generation Fails**
   - Ensure Orpheus service is running
   - Check network connectivity between services
   - Verify output directory permissions

3. **Slow Response Times**
   - Medical AI responses can take 30-60 seconds
   - Consider reducing max_length parameter
   - Monitor resource usage

### Performance Tips
- Use lower temperature (0.3-0.5) for more focused responses
- Limit max_length for faster generation
- Disable audio generation for faster text-only responses

## 📊 Expected Response Times
- **Meditron text generation**: 30-60 seconds
- **Orpheus audio synthesis**: 5-15 seconds
- **Total pipeline**: 45-75 seconds