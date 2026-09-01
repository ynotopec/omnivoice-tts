#!/usr/bin/env python3
"""OmniVoice TTS server — OpenAI-compatible TTS API with voice cloning support."""

import os
import io
import re
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
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

# CORS (allow browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model instance ---
model = None
model_device_map = "cuda:0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    model_device_map = "cpu"


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


def normalize_name(name: str) -> str:
    """Normalize voice name for matching: lowercase, strip spaces/special chars."""
    return re.sub(r'[^a-z0-9]+', '', name.lower())


def load_voices():
    """Scan voices_dir for reference audio files.
    Match audio files to .txt files by comparing normalized names."""
    voices = {}
    voices_path = Path(VOICES_DIR)
    if not voices_path.exists():
        logger.warning(f"Voices directory {VOICES_DIR} does not exist, creating it.")
        voices_path.mkdir(parents=True, exist_ok=True)
        return voices

    # Collect all audio files
    audio_files = []
    txt_files = set()

    for f in sorted(voices_path.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
            audio_files.append(f)
        elif f.suffix.lower() == ".txt":
            txt_files.add(f.stem)

    for f in audio_files:
        audio_name = f.stem  # e.g. "Marco-Voix 260901_203931" or "marco-voice"
        norm_audio = normalize_name(audio_name)

        # Try exact match first (same stem + .txt)
        ref_text_path = f.with_suffix(".txt")
        ref_text = ""

        if ref_text_path.exists():
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        else:
            # Try fuzzy match: find a .txt whose normalized name is a substring of or contains the audio's normalized name
            for txt_name in txt_files:
                norm_txt = normalize_name(txt_name)
                if len(norm_txt) >= 4 and (norm_txt in norm_audio or norm_audio in norm_txt):
                    ref_text_path = voices_path / f"{txt_name}.txt"
                    if ref_text_path.exists():
                        ref_text = ref_text_path.read_text(encoding="utf-8").strip()
                        break

        if ref_text:
            voice_id = normalize_name(audio_name)
        else:
            voice_id = audio_name

        voices[voice_id] = (str(f), ref_text)
        logger.info(f"Loaded voice '{voice_id}' from {f} (ref_text={ref_text!r})")
    return voices


# Voices registry: {voice_id: (ref_audio_path, ref_text)}
available_voices = {}


@app.on_event("startup")
def startup():
    """Pre-load model on startup (if GPU available)."""
    load_model()
    global available_voices
    available_voices = load_voices()
    logger.info(f"Available voices: {list(available_voices.keys())}")


# --- Authentication middleware ---
@app.middleware("http")
async def authenticate(request: Request, call_next):
    """Check API key on /v1/* endpoints."""
    if not API_KEY:
        return await call_next(request)

    path = request.url.path
    # Only protect /v1/* endpoints
    if not path.startswith("/v1/"):
        return await call_next(request)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        # Check query param
        token = request.query_params.get("api_key", "")

    if not token or token != API_KEY:
        return Response(
            content='{"detail":"Invalid or missing API key"}',
            status_code=401,
            media_type="application/json",
        )

    return await call_next(request)


# --- Health ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE,
        "voices": len(available_voices),
    }


# --- OpenAI-compatible endpoints ---
@app.get("/v1/models")
def list_models():
    """Return models endpoint (OpenAI compatible)."""
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
def list_voices():
    """Return list of available cloned voices."""
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

            kwargs["ref_audio"] = ref_audio_path
            kwargs["ref_text"] = request.reference_text or ""

        elif request.voice in available_voices:
            ref_audio, ref_text = available_voices[request.voice]
            kwargs["ref_audio"] = ref_audio
            kwargs["ref_text"] = ref_text
        elif request.voice != "default":
            # Try fuzzy match on normalized name
            norm_requested = normalize_name(request.voice)
            matched = None
            for vid, (path, txt) in available_voices.items():
                if norm_requested in vid or vid in norm_requested:
                    matched = (path, txt)
                    break
            if matched:
                kwargs["ref_audio"] = matched[0]
                kwargs["ref_text"] = matched[1]
            # else: auto voice, no cloning

        # Generate audio
        audio = m.generate(**kwargs)

        if not audio:
            raise HTTPException(status_code=500, detail="Model returned no audio")

        waveform = audio[0] if isinstance(audio, list) else audio

        # Apply speed
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
            sound = pydub.AudioSegment(
                waveform.tobytes(),
                frame_rate=24000,
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


# --- Voice management ---
@app.post("/v1/voices/register")
async def register_voice(request: VoiceRegistrationRequest):
    """Register a new voice from base64-encoded reference audio."""
    try:
        audio_bytes = base64.b64decode(request.reference_audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

    voice_dir = Path(VOICES_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)

    audio_path = voice_dir / f"{request.name}.wav"
    audio_path.write_bytes(audio_bytes)

    text_path = voice_dir / f"{request.name}.txt"
    text_path.write_text(request.reference_text, encoding="utf-8")

    voice_id = normalize_name(request.name)
    available_voices[voice_id] = (str(audio_path), request.reference_text)
    logger.info(f"Registered voice '{voice_id}'")

    return {"status": "ok", "voice": voice_id}


@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    """Remove a registered voice."""
    voice_dir = Path(VOICES_DIR)
    # Match by normalized name
    matched = None
    for vid in available_voices:
        if normalize_name(vid) == normalize_name(voice_name) or vid == voice_name:
            matched = vid
            break

    if matched:
        available_voices.pop(matched, None)
        audio_path = voice_dir / f"{matched}.wav"
        text_path = voice_dir / f"{matched}.txt"
        if audio_path.exists():
            audio_path.unlink()
        if text_path.exists():
            text_path.unlink()

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
