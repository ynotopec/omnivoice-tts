#!/usr/bin/env python3
"""OmniVoice TTS server — OpenAI-compatible TTS API with voice cloning support."""

import os
import io
import re
import base64
import logging
import secrets
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(os.path.expanduser("~"), ".omnivoice", "voices"))
HOST = os.environ.get("OMNIVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("OMNIVOICE_PORT", "8001"))
API_TOKEN = os.environ.get("OMNIVOICE_API_TOKEN", os.environ.get("OMNIVOICE_API_KEY", ""))
SAMPLE_RATE = 24000


def env_bool(name: str, default: bool = True) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


OMNIVOICE_DENOISE = env_bool("OMNIVOICE_DENOISE")
OMNIVOICE_PREPROCESS_PROMPT = env_bool("OMNIVOICE_PREPROCESS_PROMPT")
OMNIVOICE_POSTPROCESS_OUTPUT = env_bool("OMNIVOICE_POSTPROCESS_OUTPUT")

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
    allow_credentials=False,
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


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert decoded/model audio to a finite mono float32 waveform."""
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio)
    audio = np.squeeze(audio)
    if audio.ndim == 2:
        # soundfile returns (samples, channels), while models commonly return
        # (channels, samples). Treat the smaller dimension as channels.
        audio = audio.mean(axis=1 if audio.shape[1] <= audio.shape[0] else 0)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must contain a non-empty mono waveform")
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0)


def _decode_audio(source: Union[bytes, str, Path]) -> tuple[np.ndarray, int]:
    """Decode audio without trusting a client-supplied filename extension."""
    input_value = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    try:
        audio, sample_rate = sf.read(input_value, dtype="float32", always_2d=False)
        return _as_mono_float32(audio), sample_rate
    except Exception:
        # ffmpeg/pydub adds support for formats such as m4a and some MP3 builds.
        try:
            from pydub import AudioSegment

            segment = AudioSegment.from_file(io.BytesIO(source) if isinstance(source, bytes) else str(source))
            segment = segment.set_channels(1)
            samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
            scale = float(1 << (8 * segment.sample_width - 1))
            return _as_mono_float32(samples / scale), segment.frame_rate
        except Exception as decode_error:
            raise ValueError("Unsupported or corrupt reference audio") from decode_error


def reference_input(source: Union[bytes, str, Path]):
    """Return the exact reference shape accepted by OmniVoice.generate()."""
    if not isinstance(source, bytes):
        return str(source)
    audio, sample_rate = _decode_audio(source)
    if float(np.max(np.abs(audio))) < 1e-5:
        raise ValueError("Reference audio is silent")
    # OmniVoice accepts (torch waveform, sample rate), performs its own mono
    # conversion/resampling, and applies its native prompt preprocessing.
    return torch.from_numpy(audio), sample_rate


def encode_audio(waveform, output_format: str, sample_rate: int = SAMPLE_RATE) -> tuple[io.BytesIO, str]:
    """Serialize model output safely and return its MIME type."""
    waveform = _as_mono_float32(waveform)
    fmt = output_format.lower()
    if fmt not in ("mp3", "wav", "flac"):
        raise HTTPException(status_code=400, detail="response_format must be mp3, wav, or flac")
    buffer = io.BytesIO()
    if fmt == "mp3":
        from pydub import AudioSegment

        pcm = (waveform * 32767).astype("<i2")
        AudioSegment(pcm.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1).export(
            buffer, format="mp3", bitrate="192k"
        )
    else:
        sf.write(buffer, waveform, sample_rate, format=fmt.upper(), subtype="PCM_16")
    buffer.seek(0)
    return buffer, {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}[fmt]


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
    if not API_TOKEN:
        return await call_next(request)

    path = request.url.path
    # Only protect /v1/* endpoints
    if not path.startswith("/v1/"):
        return await call_next(request)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""

    if not token or not secrets.compare_digest(token, API_TOKEN):
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
    input: str = Field(..., min_length=1, description="The text to synthesize")
    voice: str = Field("default", description="Voice name. Use 'clone' for on-the-fly cloning, or a registered voice name.")
    response_format: str = Field("mp3", description="mp3, wav, or flac")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    language: Optional[str] = Field(None, description="Language name or code, e.g. Portuguese or pt")
    normalize_text: bool = Field(False, description="Use OmniVoice text normalization")
    denoise: Optional[bool] = Field(None, description="Use OmniVoice's native <|denoise|> conditioning token")
    preprocess_prompt: Optional[bool] = Field(None, description="Use OmniVoice reference silence removal and trimming")
    postprocess_output: Optional[bool] = Field(None, description="Use OmniVoice generated-audio postprocessing")
    reference_audio: Optional[str] = Field(None, description="Base64-encoded reference audio for voice cloning")
    reference_text: Optional[str] = Field(None, description="Transcription of the reference audio")


class VoiceRegistrationRequest(BaseModel):
    name: str
    reference_audio: str  # base64
    reference_text: Optional[str] = None


# --- TTS endpoint (OpenAI compatible) ---
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate speech from text (OpenAI /v1/audio/speech compatible)."""
    try:
        m = load_model()

        kwargs = {
            "text": request.input,
            "language": request.language,
            "speed": request.speed,
            "normalize_text": request.normalize_text,
            "denoise": OMNIVOICE_DENOISE if request.denoise is None else request.denoise,
            "preprocess_prompt": (
                OMNIVOICE_PREPROCESS_PROMPT
                if request.preprocess_prompt is None
                else request.preprocess_prompt
            ),
            "postprocess_output": (
                OMNIVOICE_POSTPROCESS_OUTPUT
                if request.postprocess_output is None
                else request.postprocess_output
            ),
        }
        reference = None
        reference_text = ""
        if request.voice == "clone":
            if not request.reference_audio:
                raise HTTPException(status_code=400, detail="voice='clone' requires reference_audio")
            try:
                reference = base64.b64decode(request.reference_audio, validate=True)
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail="Invalid reference_audio base64") from e
            reference_text = request.reference_text or None
        elif request.voice in available_voices:
            reference, reference_text = available_voices[request.voice]
        elif request.voice != "default":
            norm_requested = normalize_name(request.voice)
            matches = [value for vid, value in available_voices.items() if norm_requested == normalize_name(vid)]
            if not matches:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")
            reference, reference_text = matches[0]

        if reference is not None:
            kwargs["ref_audio"] = reference_input(reference)
            # None activates OmniVoice's documented Whisper auto-transcription;
            # an empty string would incorrectly suppress it.
            kwargs["ref_text"] = reference_text or None
        audio = m.generate(**kwargs)

        if audio is None or (hasattr(audio, "__len__") and len(audio) == 0):
            raise HTTPException(status_code=500, detail="Model returned no audio")

        waveform = audio[0] if isinstance(audio, list) else audio

        buffer, content_type = encode_audio(waveform, request.response_format, m.sampling_rate)

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
        audio_bytes = base64.b64decode(request.reference_audio, validate=True)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from e

    voice_id = normalize_name(request.name)
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice name must contain letters or numbers")
    voice_dir = Path(VOICES_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)

    audio_path = voice_dir / f"{voice_id}.wav"
    try:
        audio, sample_rate = _decode_audio(audio_bytes)
        if float(np.max(np.abs(audio))) < 1e-5:
            raise ValueError("Reference audio is silent")
        # Store decoded PCM without altering the conditioning signal. OmniVoice
        # owns resampling, level normalization, silence removal, and trimming.
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    text_path = voice_dir / f"{voice_id}.txt"
    reference_text = request.reference_text.strip() if request.reference_text else ""
    if reference_text:
        text_path.write_text(reference_text, encoding="utf-8")
    else:
        text_path.unlink(missing_ok=True)

    available_voices[voice_id] = (str(audio_path), reference_text or None)
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
    language: Optional[str] = Form(None),
):
    """Quick TTS with voice upload in a single call."""
    try:
        m = load_model()

        audio_data = await reference_audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio uploaded")

        audio = m.generate(
            text=text,
            language=language,
            ref_audio=reference_input(audio_data),
            ref_text=reference_text or None,
            denoise=OMNIVOICE_DENOISE,
            preprocess_prompt=OMNIVOICE_PREPROCESS_PROMPT,
            postprocess_output=OMNIVOICE_POSTPROCESS_OUTPUT,
        )
        waveform = audio[0] if isinstance(audio, list) else audio
        buffer, content_type = encode_audio(waveform, output_format, m.sampling_rate)

        return StreamingResponse(buffer, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clone TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
