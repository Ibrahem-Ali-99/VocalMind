# VocalMind Kaggle Inference API

High-performance AI inference server hosted on Kaggle, exposed via Ngrok. This server provides Speech-to-Text (Whisper), Emotion Recognition, and Speaker Diarization (Pyannote).

**Base URL:** `https://etta-cleistogamous-untangentially.ngrok-free.dev`  
**Required Header:** `ngrok-skip-browser-warning: true` (Bypasses the ngrok interstitial page)

---

## Endpoints

### 1. Health Check (`GET /health`)
Verify the server is running and reachable. Returns `{"status": "ok"}`.

### 2. Transcription (`POST /transcribe`)
Converts audio to text using OpenAI Whisper. Returns full text, detected language, and timestamped segments.

**Example Response:**
```json
{
  "text": "Hello, thank you for calling support...",
  "language": "en",
  "segments": [{"start": 0.0, "end": 4.72, "text": " Hello, thank you..."}]
}
```

### 3. Emotion Recognition (`POST /emotion`)
Identifies the primary emotion in the audio. Returns the top emotion and a full probability distribution.

### 4. Speaker Diarization (`POST /diarize`)
Identifies unique speakers and their corresponding time ranges using Pyannote.

**Note:** If the audio has complex background noise or overlaps, it may return empty segments while still returning 200 OK.

### 5. Full Pipeline (`POST /full`)
A power-user endpoint that runs Transcription, Diarization, and Emotion recognition in parallel (server-side).

**Special Feature:** Diarization results are automatically mapped to transcription segments. Each segment in the `segments` list will contain a `speaker` field (e.g., `SPEAKER_00`, `SPEAKER_01`, or `UNKNOWN`).

---

## Example Usage (Telecom Call)

```bash
# Full analysis of a 3-minute telecom call
curl -X POST https://etta-cleistogamous-untangentially.ngrok-free.dev/full \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@telecom_call.mp3"
```

---

## Implementation Details

- **Input Format:** `multipart/form-data` with a `file` field.
- **Auto-Processing:** Stereo files are automatically mixed down to mono for Diarization processing to ensure high accuracy.
- **Concurrency:** The server uses an internal GPU lock and `asyncio.to_thread` to ensure stable processing and non-blocking health checks.
- **Detailed Schema:** See [api_response_schema.json](api_response_schema.json) for raw response objects.
