# OmniVoice TTS

OpenAI-compatible TTS API powered by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — 600+ languages, voice cloning, and voice design.

## Features

- **OpenAI-compatible API** — `/v1/audio/speech` endpoint
- **Voice cloning** — register a voice from reference audio, or clone on-the-fly
- **600+ languages** supported
- **Voice design** — speaker attributes (gender, age, pitch, dialect)
- **GPU-accelerated** inference (CUDA 12.8+)

## Quick Start

### 1. Install

```bash
cd ~/work/omnivoice-tts
bash setup.sh
```

### 2. Place reference voices

Put reference audio files and matching text files in:

```
~/.omnivoice/voices/
├── AntonioPacheco.wav
└── AntonioPacheco.txt   # "Transcription of the reference audio."
```

### 3. Start the service

```bash
bash service.sh start
# or: systemctl --user enable --now omnivoice-tts.service
```

### 4. Use the API

```bash
# List models
curl http://localhost:8000/v1/models

# Generate speech (using a registered voice)
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Olá, isto é um teste de síntese de voz.",
    "voice": "AntonioPacheco",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3

# Register a new voice (base64 audio)
curl http://localhost:8000/v1/voices/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nova_voz",
    "reference_audio": "<base64-encoded-audio>",
    "reference_text": "Transcrição do áudio de referência."
  }'

# On-the-fly voice cloning
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello world!",
    "voice": "clone",
    "reference_audio": "<base64-encoded-ref-audio>",
    "reference_text": "Reference transcription.",
    "response_format": "wav"
  }'
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List models (OpenAI compatible) |
| GET | `/v1/voices` | List registered voices |
| POST | `/v1/audio/speech` | Generate speech |
| POST | `/v1/voices/register` | Register a voice |
| DELETE | `/v1/voices/{name}` | Delete a voice |
| POST | `/v1/audio/clone` | Quick TTS with voice upload |

### Speech Request

```json
{
  "model": "omnivoice",
  "input": "Text to synthesize",
  "voice": "AntonioPacheco",  // or "clone" for on-the-fly
  "response_format": "mp3",   // mp3, wav, flac
  "speed": 1.0,               // 0.25 - 4.0
  "reference_audio": "<base64>"  // optional, for voice cloning
}
```

### Voice Registration

```json
{
  "name": "my_voice",
  "reference_audio": "<base64-encoded-wav>",
  "reference_text": "Transcription of the reference audio."
}
```

## Service Management

```bash
bash service.sh start       # Start
bash service.sh stop        # Stop
bash service.sh restart     # Restart
bash service.sh status      # Status
bash service.sh logs        # Follow logs
```

Or directly with systemd:

```bash
systemctl --user daemon-reload
systemctl --user enable --now omnivoice-tts.service
systemctl --user status omnivoice-tts.service
journalctl --user -u omnivoice-tts.service -f
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIVOICE_HOST` | `0.0.0.0` | Bind address |
| `OMNIVOICE_PORT` | `8000` | Port |
| `MODEL_NAME` | `k2-fsa/OmniVoice` | HuggingFace model |
| `MODEL_CACHE_DIR` | `~/.cache/huggingface` | Model cache location |
| `VOICES_DIR` | `~/.omnivoice/voices` | Voice reference files |

## License

OmniVoice — check the model card on HuggingFace for licensing details.
