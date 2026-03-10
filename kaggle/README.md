# VocalMind Inference API

The Kaggle Ngrok Server exposes 4 endpoints. All accept `multipart/form-data` with an audio `file`.

**Base URL:** `https://etta-cleistogamous-untangentially.ngrok-free.dev`
**Header:** `ngrok-skip-browser-warning: true` (Required)

---

### Endpoints & Examples

#### 1. Transcription (`/transcribe`)
Returns Whisper-transcribed text, language, and timestamped segments.

```bash
curl -X POST https://etta-cleistogamous-untangentially.ngrok-free.dev/transcribe \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@g:/projects/VocalMind/research/voices-examples/DEX_channel_separated_callcenter/2077589677/2077589677_final_stereo.wav"
```

#### 2. Emotion (`/emotion`)
Returns the dominant emotion and probabilities.

```bash
curl -X POST https://etta-cleistogamous-untangentially.ngrok-free.dev/emotion \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@g:/projects/VocalMind/research/voices-examples/DEX_channel_separated_callcenter/2077593127/2077593127_final_stereo.wav"
```

#### 3. Diarization (`/diarize`)
Returns Pyannote speakers timestamps (`SPEAKER_00`, `SPEAKER_01`).

```bash
curl -X POST https://etta-cleistogamous-untangentially.ngrok-free.dev/diarize \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@g:/projects/VocalMind/research/voices-examples/DEX_channel_separated_callcenter/2077589677/2077589677_final_stereo.wav"
```

#### 4. Full Pipeline (`/full`)
Combines STT, Emotion, and Diarization into one JSON output.

```bash
curl -X POST https://etta-cleistogamous-untangentially.ngrok-free.dev/full \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@g:/projects/VocalMind/research/voices-examples/DEX_channel_separated_callcenter/2077593127/2077593127_final_stereo.wav"
```
