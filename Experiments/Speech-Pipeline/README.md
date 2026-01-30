# VocalMind Speech Pipeline

**Version:** 2.0  
**Last Updated:** January 30, 2026

---

## Quick Start

```bash
# 1. Navigate to this directory
cd Experiments/Speech-Pipeline

# 2. Run the pipeline on your audio file
python speech_pipeline.py --audio your_call.mp3

# 3. Add emotion analysis
python emotion_analysis.py --audio your_call.mp3

# 4. Or run the full demo
python run_demo.py
```

### What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| **speech_pipeline.py** | Core pipeline - diarization + transcription | Speaker roles, timestamps, transcripts, confidence scores |
| **emotion_analysis.py** | Pipeline + emotion recognition | Everything above + text/audio emotions, sentiment analysis |
| **run_demo.py** | End-to-end demo | Complete workflow from audio to annotated conversation |

---

## Overview

The VocalMind Speech Pipeline is a production-ready audio processing system for call center analytics. It combines state-of-the-art speech recognition, speaker diarization, and intelligent role detection to produce annotated conversation logs.

---

## Features

- **Speaker Diarization** - Neural speaker identification using Pyannote 3.1
- **Speech Recognition** - faster-whisper with word-level timestamps and confidence scores
- **Language Detection** - Automatic language identification
- **Overlap Detection** - Identifies simultaneous speech between speakers
- **Smart Role Detection** - Automatically identifies Agent vs Customer roles
- **Text Post-Processing** - Cleans up ASR output (number formatting, spacing)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: Audio File                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYANNOTE SPEAKER DIARIZATION (3.1)                             │
│  • Neural speaker embeddings                                    │
│  • Identifies unique speakers                                   │
│  • Outputs: [(speaker_id, start_time, end_time), ...]          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASTER-WHISPER ASR                                             │
│  • Transcribes audio to text                                    │
│  • Word-level timestamps for precise alignment                  │
│  • Confidence scores per word                                   │
│  • Auto language detection                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  SPEAKER-WORD ALIGNMENT                                         │
│  • Match each word to speaker based on timestamp midpoint       │
│  • Group consecutive words by same speaker                      │
│  • Merge trailing single words to appropriate speaker           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  OVERLAP DETECTION                                              │
│  • Detect simultaneous speech                                   │
│  • Calculate overlap duration and percentage                    │
│  • Flag affected utterances                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  ROLE DETECTION                                                 │
│  • Keyword-based scoring (Agent vs Customer)                    │
│  • First-speaker heuristic                                      │
│  • Speaking pattern analysis                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Annotated Conversation Log                             │
│  • Speaker roles                                                │
│  • Transcribed text with timestamps                             │
│  • Confidence scores                                            │
│  • Overlap flags                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **Speaker Diarization** | `pyannote/speaker-diarization-3.1` | Neural speaker identification using voice embeddings |
| **Speech Recognition** | `faster-whisper` (medium/large-v3) | Speech-to-text with word timestamps |

---

## Alignment Algorithm

### Midpoint Matching

For each word, the algorithm checks which speaker segment contains its midpoint:

```python
def get_speaker_at_time(t):
    # 1. Check if t is within any segment
    for seg in speaker_segments:
        if seg['start'] <= t <= seg['end']:
            return seg['speaker']
    
    # 2. If in gap, find closest segment
    min_dist = float('inf')
    closest = None
    for seg in speaker_segments:
        seg_mid = (seg['start'] + seg['end']) / 2
        dist = abs(t - seg_mid)
        if dist < min_dist:
            min_dist = dist
            closest = seg['speaker']
    
    return closest

# For each word
word_mid = (word['start'] + word['end']) / 2
word_speaker = get_speaker_at_time(word_mid)
```

### Visual Example

```
Speaker Segments (Pyannote):
    SPEAKER_00: [=============================]
    SPEAKER_01:                                [====================]
Time (s):      0                           4.2 4.3               9.8

Words (Whisper):
    "Hello"     "thank"    "calling"    "Hi"      "Rajesh"
    0.0-0.5     0.52-0.82  1.32-1.8    4.5-4.9   5.0-5.6
    
Matching:
    "Hello"   → midpoint 0.25  → SPEAKER_00 → Agent
    "thank"   → midpoint 0.67  → SPEAKER_00 → Agent
    "calling" → midpoint 1.56  → SPEAKER_00 → Agent
    "Hi"      → midpoint 4.7   → SPEAKER_01 → Customer
    "Rajesh"  → midpoint 5.3   → SPEAKER_01 → Customer
```

---

## Usage

### Basic Speech Pipeline

Process an audio file to get speaker diarization and transcription:

```bash
# Navigate to the Speech-Pipeline-Tests directory
cd Experiments/Speech-Pipeline

# Process an audio file with medium model (recommended)
python speech_pipeline.py --audio path/to/your/audio.mp3

# Use large model for better accuracy (slower)
python speech_pipeline.py --audio call.mp3 --whisper-model large-v3

# Use small model for faster processing (lower accuracy)
python speech_pipeline.py --audio call.mp3 --whisper-model small
```

**Available Whisper Models:**
- `base` - Fastest, lowest accuracy (~1GB RAM)
- `small` - Fast, good accuracy (~2GB RAM)
- `medium` - **Recommended** - Best balance (~5GB RAM)
- `large-v3` - Best accuracy, slowest (~10GB RAM)

### Speech Pipeline with Emotion Recognition

Add emotion analysis to each utterance:

```bash
# Basic usage - analyzes both text and audio emotions
python emotion_analysis.py --audio call.mp3

# With specific Whisper model
python emotion_analysis.py --audio call.mp3 --whisper-model large-v3

# Save results to JSON file
python emotion_analysis.py --audio call.mp3 --output results.json
```

**What you get:**
- Speaker roles (Agent/Customer)
- Transcriptions with timestamps
- Text emotion for each utterance (7 classes)
- Audio emotion for each utterance (8 classes)
- Sentiment mapping (Positive/Neutral/Negative)
- Customer emotional journey tracking

### End-to-End Demo

Run the complete pipeline from generation to analysis:

```bash
# Process an existing audio file
python run_demo.py --audio telecom_call.mp3

# Generate synthetic call and process it (requires ElevenLabs API key)
python run_demo.py --generate

# Use different model
python run_demo.py --audio call.mp3 --whisper-model large-v3
```

**Note:** The `--generate` option requires an ElevenLabs API key in your `.env` file:
```
ELEVENLABS_API_KEY=your_api_key_here
```

---

## Step-by-Step Example

### Example 1: Just Transcription

You have a call recording and want to know who said what:

```bash
# Step 1: Navigate to the directory
cd Experiments/Speech-Pipeline

# Step 2: Run the pipeline
python speech_pipeline.py --audio my_customer_call.mp3
```

**You'll see:**
```
Agent (0.0s - 4.2s): Hello, thank you for calling customer support...
Customer (4.3s - 9.8s): Hi, I'm calling about my bill...
Agent (10.0s - 15.5s): I understand, let me look that up...
```

### Example 2: With Emotion Analysis

You want to understand customer sentiment during the call:

```bash
python speech_pipeline_test.py --audio my_customer_call.mp3
```

**You'll see:**
```
[1] Agent (0.0s - 4.2s)
    Text: "Hello, thank you for calling customer support..."
    Text Emotion:  😐 neutral    (72.3%) → Neutral
    Audio Emotion: 😌 calm       (68.1%) → Neutral

[2] Customer (4.3s - 9.8s)
    Text: "Hi, I'm calling about my bill..."
    Text Emotion:  🤬 anger      (85.2%) → Negative
    Audio Emotion: 🤬 angry      (71.4%) → Negative
```

### Example 3: Save Results

You want to save the analysis for later use:

```bash
python speech_pipeline_test.py --audio call.mp3 --output analysis.json
```

This creates `analysis.json` with all the data (transcripts, emotions, timestamps, etc.)

---

## Troubleshooting

### "Audio file not found"
Make sure you're in the correct directory and the path is correct:
```bash
# Use absolute path
python speech_pipeline.py --audio "C:\path\to\audio.mp3"

# Or relative path from current directory
python speech_pipeline.py --audio ../../audio.mp3
```

### "HF_TOKEN not found"
The pipeline needs access to Hugging Face models. Create a `.env` file:
```bash
# Create .env file in Speech-Pipeline-Tests directory
HF_TOKEN=your_huggingface_token
```

Get your token at: https://huggingface.co/settings/tokens

### Out of memory errors
Try a smaller Whisper model:
```bash
python speech_pipeline.py --audio call.mp3 --whisper-model small
```

---

### Python API

```python
from speech_pipeline import VocalMindPipelineV2

# Initialize pipeline
pipeline = VocalMindPipelineV2(whisper_model="medium")

# Process audio file
conversation_log = pipeline.process_file("call_recording.mp3")

# Each utterance contains:
for entry in conversation_log:
    print(f"{entry['role']}: {entry['text']}")
    print(f"  Confidence: {entry['confidence']:.2%}")
    print(f"  Timestamp: {entry['timestamp']}")
    print(f"  Has Overlap: {entry['has_overlap']}")
```

---

## Output Format

Each utterance in the conversation log contains:

```python
{
    'speaker': 'SPEAKER_00',           # Raw speaker ID
    'role': 'Agent',                   # Detected role (Agent/Customer)
    'text': 'Hello, thank you...',     # Transcribed text
    'confidence': 0.95,                # Average word confidence
    'timestamp': (0.0, 4.2),           # (start_sec, end_sec)
    'has_overlap': False,              # Overlapping speech detected
    'overlap_with': None,              # Other speaker(s) if overlapping
    'overlap_duration': 0              # Overlap duration in seconds
}
```

---

## Requirements

```
torch
transformers
faster-whisper
pyannote.audio
librosa
numpy
python-dotenv
```

### Environment Variables

Create a `.env` file with:

```
HF_TOKEN=your_huggingface_token
```

Required for Pyannote speaker diarization model access.

---

## Performance Notes

- **CPU Mode:** Uses int8 quantization for faster-whisper (compatibility)
- **GPU Mode:** Pyannote automatically uses CUDA if available
- **Memory:** Medium model ~2GB VRAM, Large-v3 ~4GB VRAM

---

## Known Limitations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Number formatting (`$2 ,500`) | Whisper tokenization | Text post-processing applied |
| Homophone errors (`waiving` → `waving`) | ASR limitation | Context-aware spell check |
| Boundary splits | Word timestamp imprecision | Trailing word merge logic |

---

## Test Script with Emotion Recognition

The `speech_pipeline_test.py` script combines the speech pipeline with emotion recognition models for full call analysis.

### Emotion Models

| Model | Modality | Classes |
|-------|----------|---------|
| `j-hartmann/emotion-english-distilroberta-base` | Text | 7: anger, disgust, fear, joy, neutral, sadness, surprise |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Audio | Dimensional (arousal, valence, dominance) → mapped to 7 emotions |

### Usage

```bash
python emotion_analysis.py --audio path/to/call.mp3
python emotion_analysis.py --audio call.mp3 --whisper-model large-v3
python emotion_analysis.py --audio call.mp3 --output results.json
```

### Test Output

The test script outputs:
1. **Conversation log** with speaker roles and transcriptions
2. **Text emotion** for each utterance (from transcript)
3. **Audio emotion** for each utterance (from audio segment)
4. **Sentiment mapping** (Positive/Neutral/Negative)
5. **Customer emotional journey** tracking

### Sample Output

```
[1] Agent (0.0s - 4.2s)
    Text: "Hello, thank you for calling customer support..."
    Text Emotion:  😐 neutral    (72.3%) → Neutral
    Audio Emotion: 😌 calm       (68.1%) → Neutral

[2] Customer (4.3s - 9.8s)
    Text: "I'm really upset about my bill..."
    Text Emotion:  🤬 anger      (85.2%) → Negative
    Audio Emotion: 🤬 angry      (71.4%) → Negative
```

### Sentiment Summary

| Text Emotion | Audio Emotion | Sentiment |
|--------------|---------------|-----------|
| anger | angry | Negative |
| disgust | disgust | Negative |
| fear | fearful | Negative |
| sadness | sad | Negative |
| joy | happy | Positive |
| surprise | surprised | Positive |
| neutral | neutral | Neutral |
| — | calm | Neutral |

---

## Known Limitations & Future Enhancements

### Current Limitations

| Issue | Description | Status |
|-------|-------------|--------|
| **Text model misclassifications** | Text emotion model trained on social media/reviews, not call center dialog. May misclassify phrases like "I understand your frustration" as anger. | Known - needs fine-tuning |
| **Audio model on synthetic speech** | ElevenLabs synthetic voices lack emotional prosody, causing less reliable predictions on generated test audio. Works better on real recordings. | Expected behavior |
| **Currency/number formatting** | ASR may split decimals (e.g., "8 .99" instead of "8.99") and mix currency formats. | Post-processing needed |
| **Phone number transcription** | Numbers often transcribed with awkward spacing or punctuation. | Post-processing needed |

### Multimodal Fusion Strategy

The current implementation uses **weighted fusion**:
- When text and audio agree → boosted confidence
- When they disagree → higher confidence wins (with audio discounted by 20% due to synthetic speech issues)

### Planned Enhancements (Enhancement Phase)

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Fine-tune text model on call center transcripts | High | Very High | P1 |
| Fine-tune audio model on real call recordings | High | Very High | P1 |
| Add role-aware emotion expectations (agents = neutral/positive) | Low | Medium | P2 |
| Implement number/currency post-processing | Low | Low | P3 |
| Batch processing for GPU efficiency | Medium | Medium | P3 |
| Ensemble multiple emotion models | Medium | High | P2 |

### Model Information

| Component | Model | Training Data | Notes |
|-----------|-------|---------------|-------|
| Text Emotion | j-hartmann/emotion-english-distilroberta-base | Social media, reviews | 7 classes |
| Audio Emotion | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | MSP-Podcast (natural speech) | Dimensional (arousal, dominance, valence) |
| ASR | faster-whisper | OpenAI Whisper | Configurable model size |
| Diarization | pyannote/speaker-diarization-3.1 | Multiple datasets | CUDA accelerated |
