# OmniVoice TTS

Minimal [OpenAI-compatible](https://platform.openai.com/docs/api-reference/audio/createSpeech)
TTS server for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice),
including multilingual voice cloning.

## Start

```bash
./install.sh                         # also upgrades an existing installation
./run.sh 0.0.0.0 8001               # PORT is chosen automatically if omitted
```

`install.sh` uses `uv`, installs Python 3.12 when needed, and creates the virtual
environment at `~/venv/$(basename "$PWD")`. It is safe to run repeatedly. Models
use the standard Hugging Face cache; the installer does not duplicate or eagerly
download them.

To run at login instead:

```bash
systemctl --user enable --now "$(basename "$PWD").service"
journalctl --user -fu "$(basename "$PWD").service"
```

The generated user service calls the same `run.sh` and is replaced safely on each
upgrade. Edit `.env` after installation to set a bearer token or override defaults.

## OpenAI speech API

```bash
TOKEN="$(sed -n 's/^OMNIVOICE_API_TOKEN=//p' .env)"
curl http://127.0.0.1:8001/v1/audio/speech \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "omnivoice",
    "input": "Olá, isto é um teste de síntese de voz.",
    "voice": "default",
    "language": "Portuguese",
    "response_format": "mp3"
  }' --output speech.mp3
```

OpenAPI documentation is available at `/docs`; `/v1/models` and `/v1/voices`
provide discovery. All `/v1/*` routes require `Authorization: Bearer <token>` when
`OMNIVOICE_API_TOKEN` is set.

For a persistent cloned voice, place a 3–10 second recording and its exact
transcript in `~/.omnivoice/voices` with matching names, for example
`portuguese.wav` and `portuguese.txt`, then restart. WAV/FLAC without music, echo,
or overlapping speech gives the cleanest result. OmniVoice performs its own native
reference resampling, prompt preprocessing, and denoise conditioning.

### Multilingual cloning limitations

OmniVoice's [upstream voice-cloning tips](https://github.com/k2-fsa/OmniVoice#voice-cloning)
explicitly note that cross-lingual cloning retains an accent from the reference
recording's language. The `language` request
field supplies useful target-language conditioning, but it cannot remove that
model limitation. For standard pronunciation, register a separate 3–10 second
reference spoken by the same person in each target language and select the
matching voice when synthesizing. For example:

```text
~/.omnivoice/voices/marco-fr.wav  + marco-fr.txt
~/.omnivoice/voices/marco-en.wav  + marco-en.txt
~/.omnivoice/voices/marco-pt.wav  + marco-pt.txt
```

The transcript must exactly match what is spoken in its audio file. Supplying an
incorrect transcript—or a translation instead of the original words—degrades
alignment, pronunciation, and speaker similarity. Omitting `reference_text`
enables OmniVoice's Whisper transcription, but an exact manual transcript is
preferable when available.

Pronunciation overrides are model-specific: English accepts bracketed CMU
phonemes such as `[B EY1 S]`, while Chinese accepts pinyin with tone numbers.
These controls are not a general cross-language accent conversion mechanism.
Likewise, upstream voice-design accent controls are trained for English, and
Chinese dialect controls are trained for Chinese; they should not be presented
as a fix for a cloned voice in another language.

## Hardware

Dependencies are intentionally not tied to an x86-only CUDA wheel index. `uv`
resolves the PyTorch build exposed for the host platform, allowing the same install
flow on NVIDIA H100 systems and ARM64 NVIDIA DGX Spark. If a platform image provides
a vendor PyTorch build, activate the created environment and install that build
before rerunning `./install.sh`. MP3 input/output additionally requires `ffmpeg` on
`PATH`; WAV and FLAC do not.

See [`.env.example`](.env.example) for the small set of deployment variables.
