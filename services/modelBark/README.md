# Bark TTS Service

Alternative Text-to-Speech service using Suno AI's Bark model.

## Features

- **Multilingual support** - English, Spanish, French, German, Chinese, etc.
- **Multiple voice presets** - 10 English voices + international voices  
- **High quality** - Natural sounding speech with emotional tones
- **Open source** - No API keys needed

## API Endpoints

### Health Check
```bash
curl http://localhost:5008/health
```

### Synthesize Speech
```bash
curl -X POST http://localhost:5008/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is Bark TTS speaking!",
    "voice": "v2/en_speaker_6"
  }'
```

### List Available Voices
```bash
curl http://localhost:5008/voices
```

## Voice Presets

**English:**
- `v2/en_speaker_0` through `v2/en_speaker_9`

**Other Languages:**
- `v2/de_speaker_0` (German)
- `v2/es_speaker_0` (Spanish)  
- `v2/fr_speaker_0` (French)
- `v2/zh_speaker_0` (Chinese)
- And more!

## Architecture

- **Port:** 5008
- **Output directory:** `../../outputs/bark/`
- **Model cache:** `~/.cache/suno/bark_v0/`

## Comparison with Orpheus

| Feature | Orpheus | Bark |
|---------|---------|------|
| Port | 5005 | 5008 |
| Model size | ~3B parameters | Mixed (1.3B text, 350M audio) |
| Voices | 1 (Tara) | 10+ (multilingual) |
| Quality | Medical-focused | General purpose |
| Setup | Requires HF token | No authentication |
| Languages | English | 13+ languages |

## Notes

- First run downloads ~5GB of models (can take 5-10 minutes)
- Models are cached in `~/.cache/suno/bark_v0/`
- Works entirely offline after initial download
- GPU recommended but CPU works (slower)

## Integration

Both Orpheus and Bark run simultaneously. Choose based on your needs:
- **Orpheus:** Medical domain, single consistent voice
- **Bark:** Multiple voices, multilingual, general purpose
