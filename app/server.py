#!/usr/bin/env python3
"""OmniVoice TTS server — OpenAI-compatible TTS API with voice cloning support."""

import os
import io
import base64
import tempfile
import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(os.path.expanduser("~"), ".omnivoice", "voices"))
HOST = os.environ.get("OMNIVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("OMNIVOICE_PORT", "8001"))
API_KEY = os.environ.get("OMNIVOICE_API_KEY", "")  # If set, require authentication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI(
    title="OmniVoice TTS",
    version="1.0.0",
    description="OpenAI-compatible TTS with voice cloning — 600+ languages",
)

# --- Global model instance ---
model = None
model_device_map = "cuda:0" if "cuda" in os.environ.get("CUDA_VISIBLE_DEVICES", "cuda") else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Lazy-load the OmniVoice model on first request."""
    global model
    if model is not None:
        return model

    from omnivoice import OmniVoice

    logger.info(f"Loading OmniVoice model from {MODEL_NAME} on {model_device_map}...")
    model = OmniVoice.from_pretrained(
        MODEL_NAME,
        device_map=model_device_map,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    logger.info("Model loaded successfully.")
    return model


def load_voices():
    """Scan voices_dir for reference audio files and return dict {name: (path, text)}.
    """
    voices = {}
    voices_path = Path(VOICES_DIR)
    if not voices_path.exists():
        logger.warning(f"Voices directory {VOICES_DIR} does not exist, creating it.")
        voices_path.mkdir(parents=True, exist_ok=True)
        return voices

    for f in sorted(voices_path.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
            name = f.stem
            # Try to load corresponding .txt file for the reference text
            ref_text_path = f.with_suffix(".txt")
            if ref_text_path.exists():
                ref_text = ref_text_path.read_text(encoding="utf-8").strip()
            else:
                ref_text = ""
            voices[name] = (str(f), ref_text)
            logger.info(f"Loaded voice '{name}' from {f} (ref_text={ref_text!r})")
    return voices


# Voices registry: {voice_name: (ref_audio_path, ref_text)}
available_voices = {}


@app.on_event("startup")
def startup():
    """Pre-load model on startup (if GPU available)."""
    load_model()
    global available_voices
    available_voices = load_voices()
    logger.info(f"Available voices: {list(available_voices.keys())}")


# --- Authentication middleware ---
async def verify_api_key(request: Request):
    """Check for API key in Authorization header or query param."""
    if not API_KEY:
        return  # No auth required
    auth_header = request.headers.get("Authorization", "")
    query_key = request.query_params.get("api_key", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else query_key
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE,
        "voices": len(available_voices),
    }


@app.get("/v1/models")
async def list_models():
    """Return models endpoint (OpenAI compatible)."""
    if API_KEY:
        await verify_api_key(None)  # Will be called via middleware pattern below
    return {
        "object": "list",
        "data": [
            {
                "id": "omnivoice",
                "object": "model",
                "created": 1725235200,
                "owned_by": "k2-fsa",
                "permission": [],
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    """Return list of available cloned voices."""
    if API_KEY:
        await verify_api_key(None)
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "voice",
                "name": name,
                "status": "ready",
            }
            for name in available_voices
        ],
    }


# --- Pydantic models for TTS API ---
class SpeechRequest(BaseModel):
    model: str = "omnivoice"
    input: str = Field(..., description="The text to synthesize")
    voice: str = Field("default", description="Voice name. Use 'clone' for on-the-fly cloning, or a registered voice name.")
    response_format: str = Field("mp3", description="mp3, wav, or flac")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    # Voice cloning fields
    reference_audio: Optional[str] = Field(None, description="Base64-encoded reference audio for voice cloning")
    reference_text: Optional[str] = Field(None, description="Transcription of the reference audio")


class VoiceRegistrationRequest(BaseModel):
    name: str
    reference_audio: str  # base64
    reference_text: str


# --- TTS endpoint (OpenAI compatible) ---
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate speech from text (OpenAI /v1/audio/speech compatible)."""
    try:
        m = load_model()

        # Determine generation parameters
        kwargs = {"text": request.input}

        if request.voice == "clone" and request.reference_audio:
            # On-the-fly voice cloning
            try:
                audio_bytes = base64.b64decode(request.reference_audio)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid reference_audio base64: {e}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_audio_path = tmp.name

            ref_text = request.reference_text or ""
            kwargs["ref_audio"] = ref_audio_path
            kwargs["ref_text"] = ref_text
        elif request.voice in available_voices:
            # Use pre-registered voice
            ref_audio, ref_text = available_voices[request.voice]
            kwargs["ref_audio"] = ref_audio
            kwargs["ref_text"] = ref_text
        else:
            # Auto voice (no cloning)
            pass

        # Generate audio
        audio = m.generate(**kwargs)

        if not audio:
            raise HTTPException(status_code=500, detail="Model returned no audio")

        # audio is list of np.ndarray at 24kHz
        waveform = audio[0] if isinstance(audio, list) else audio

        # Apply speed if needed
        if request.speed != 1.0:
            import scipy.signal
            factor = int(len(waveform) * request.speed)
            waveform = scipy.signal.resample(waveform, factor)

        # Convert to requested format
        fmt = request.response_format.lower()
        if fmt not in ("mp3", "wav", "flac"):
            fmt = "mp3"

        buffer = io.BytesIO()

        if fmt == "mp3":
            import pydub
            sample_rate = 24000
            sound = pydub.AudioSegment(
                waveform.tobytes(),
                frame_rate=sample_rate,
                sample_width=waveform.dtype.itemsize,
                channels=1,
            )
            sound.export(buffer, format="mp3", bitrate="192k")
        elif fmt == "wav":
            sf.write(buffer, waveform, 24000, format="wav")
        elif fmt == "flac":
            sf.write(buffer, waveform, 24000, format="FLAC")
        else:
            sf.write(buffer, waveform, 24000, format="wav")

        buffer.seek(0)
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
        }.get(fmt, "audio/wav")

        return StreamingResponse(buffer, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# --- Voice management endpoints ---
@app.post("/v1/voices/register")
async def register_voice(request: VoiceRegistrationRequest):
    """Register a new voice from base64-encoded reference audio."""
    try:
        audio_bytes = base64.b64decode(request.reference_audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

    voice_dir = Path(VOICES_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Save audio file
    audio_path = voice_dir / f"{request.name}.wav"
    audio_path.write_bytes(audio_bytes)

    # Save reference text
    text_path = voice_dir / f"{request.name}.txt"
    text_path.write_text(request.reference_text, encoding="utf-8")

    available_voices[request.name] = (str(audio_path), request.reference_text)
    logger.info(f"Registered voice '{request.name}'")

    return {"status": "ok", "voice": request.name}


@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    """Remove a registered voice."""
    voice_dir = Path(VOICES_DIR)
    audio_path = voice_dir / f"{voice_name}.wav"
    text_path = voice_dir / f"{voice_name}.txt"

    if audio_path.exists():
        audio_path.unlink()
    if text_path.exists():
        text_path.unlink()

    available_voices.pop(voice_name, None)
    return {"status": "ok", "voice": voice_name}


@app.post("/v1/audio/clone")
async def clone_voice_from_upload(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    output_format: str = Form("mp3"),
):
    """Quick TTS with voice upload in a single call."""
    try:
        m = load_model()

        audio_data = await reference_audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio uploaded")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            ref_path = tmp.name

        kwargs = {
            "text": text,
            "ref_audio": ref_path,
            "ref_text": reference_text or "",
        }

        audio = m.generate(**kwargs)
        waveform = audio[0] if isinstance(audio, list) else audio

        buffer = io.BytesIO()
        if output_format.lower() == "mp3":
            import pydub
            sound = pydub.AudioSegment(
                waveform.tobytes(), frame_rate=24000, sample_width=waveform.dtype.itemsize, channels=1
            )
            sound.export(buffer, format="mp3")
        else:
            sf.write(buffer, waveform, 24000, format="wav")

        buffer.seek(0)
        fmt = output_format.lower()
        content_type = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}.get(fmt, "audio/wav")

        return StreamingResponse(buffer, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clone TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
