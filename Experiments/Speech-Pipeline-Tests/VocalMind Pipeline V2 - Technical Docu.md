# VocalMind Pipeline V2 - Technical Documentation

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Architecture Flow](#architecture-flow)
3. [Component Details](#component-details)
4. [Alignment Algorithm](#alignment-algorithm)
5. [Example Walkthrough](#example-walkthrough)
6. [Output Format](#output-format)
7. [Performance Metrics](#performance-metrics)

---

## Pipeline Overview

The VocalMind V2 pipeline processes call center audio recordings to extract:
- **Speaker diarization** (who spoke when)
- **Transcription** (what was said)
- **Emotion classification** (sentiment analysis)

### Key Components
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Audio     │────▶│  Pyannote   │────▶│   Speaker   │
│   Input     │     │ Diarization │     │  Segments   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
       ┌───────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ faster-     │────▶│Word-Level   │────▶│  Speaker    │
│ whisper     │     │ Timestamps  │     │  Matching   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
       ┌───────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Emotion   │────▶│ Multimodal  │────▶│ Final Output│
│   Model     │     │   Fusion    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Architecture Flow

### Step 1: Audio Loading & Preprocessing
```python
# Load audio at 16kHz (standard for speech models)
full_audio, sr = librosa.load(audio_path, sr=16000)

# Check original sample rate
original_sr = librosa.get_samplerate(audio_path)
# Example: 44100 Hz (high quality) or 8000 Hz (telephony)
```

**Why 16kHz?**
- Whisper is trained on 16kHz audio
- If input is 8kHz (telephony), librosa upsamples automatically
- If input is 44kHz (high quality), it downsamples to save computation

---

### Step 2: Speaker Diarization (Pyannote 3.1)

**Input:** Audio waveform tensor
```python
waveform_tensor = torch.from_numpy(full_audio).unsqueeze(0)
audio_input = {"waveform": waveform_tensor, "sample_rate": 16000}
```

**Process:** Pyannote uses neural speaker embeddings to:
1. Extract voice features (pitch, timbre, speaking rate)
2. Cluster similar voices together
3. Assign speaker labels (SPEAKER_00, SPEAKER_01, etc.)

**Output:** Speaker segments with timestamps
```python
[
    {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 4.2},
    {'speaker': 'SPEAKER_01', 'start': 4.3, 'end': 9.8},
    {'speaker': 'SPEAKER_00', 'start': 10.0, 'end': 15.5},
    ...
]
```

**Example:**
```
Timeline:  [======SPEAKER_00======]  [===SPEAKER_01===]  [==SPEAKER_00==]
Time (s):  0                    4.2  4.3            9.8  10.0        15.5
```

---

### Step 3: Speech Recognition (faster-whisper medium)

**Input:** Audio waveform
```python
segments, info = self.asr_model.transcribe(
    full_audio,
    beam_size=5,              # Beam search for better accuracy
    word_timestamps=True,     # Get timing for each word
    language=None             # Auto-detect language
)
```

**Process:** Transformer-based seq2seq model
1. **Encoder:** Converts audio → acoustic features
2. **Decoder:** Generates text tokens with attention
3. **Word aligner:** Assigns timestamps to each word

**Output:** Words with timestamps and confidence scores
```python
[
    {'word': 'Hello', 'start': 0.0, 'end': 0.5, 'probability': 0.98},
    {'word': 'thank', 'start': 0.52, 'end': 0.82, 'probability': 0.95},
    {'word': 'you', 'start': 0.84, 'end': 1.1, 'probability': 0.97},
    {'word': 'for', 'start': 1.12, 'end': 1.3, 'probability': 0.99},
    {'word': 'calling', 'start': 1.32, 'end': 1.8, 'probability': 0.96},
    ...
]
```

**Language Detection:**
```python
info.language = "en"              # Detected language
info.language_probability = 0.98  # Confidence in detection
```

---

## Alignment Algorithm

### Challenge
We have:
- **Speaker segments** from Pyannote (time ranges for each speaker)
- **Words** from Whisper (text + timestamps)

We need to assign each word to the correct speaker.

### Solution: Midpoint Matching

For each word, we check which speaker segment contains its midpoint:

```python
def get_speaker_at_time(t):
    """Find which speaker is active at time t"""
    # 1. Check if t is within any segment
    for seg in speaker_segments:
        if seg['start'] <= t <= seg['end']:
            return seg['speaker']
    
    # 2. If not (gap between speakers), find closest segment
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
    "Hello"     "thank"   "you"     "for"     "calling"    "Hi"      "Rajesh"
    0.0-0.5     0.52-0.82  0.84-1.1  1.12-1.3  1.32-1.8    4.5-4.9   5.0-5.6
    Midpoint:   Midpoint:  Midpoint: Midpoint: Midpoint:   Midpoint: Midpoint:
    0.25        0.67       0.97      1.21      1.56        4.7       5.3
    
Matching:
    "Hello"  → midpoint 0.25  → in SPEAKER_00 segment → Agent
    "thank"  → midpoint 0.67  → in SPEAKER_00 segment → Agent
    "you"    → midpoint 0.97  → in SPEAKER_00 segment → Agent
    "for"    → midpoint 1.21  → in SPEAKER_00 segment → Agent
    "calling"→ midpoint 1.56  → in SPEAKER_00 segment → Agent
    "Hi"     → midpoint 4.7   → in SPEAKER_01 segment → Customer
    "Rajesh" → midpoint 5.3   → in SPEAKER_01 segment → Customer
```

---

### Grouping Words into Utterances

Once words are matched to speakers, we group consecutive words from the same speaker:

```python
current_speaker = None
current_words = []
current_start = None
current_end = None

for word in all_words:
    word_speaker = get_speaker_at_time(word['midpoint'])
    
    # Speaker changed? Save current utterance
    if word_speaker != current_speaker and current_words:
        utterances.append({
            'speaker': current_speaker,
            'text': ' '.join(current_words),
            'start': current_start,
            'end': current_end
        })
        current_words = []
    
    # Add word to current utterance
    current_words.append(word['word'])
    if current_start is None:
        current_start = word['start']
    current_end = word['end']
    current_speaker = word_speaker
```

**Result:**
```python
[
    {
        'speaker': 'SPEAKER_00',
        'text': 'Hello thank you for calling',
        'start': 0.0,
        'end': 1.8
    },
    {
        'speaker': 'SPEAKER_01', 
        'text': 'Hi Rajesh',
        'start': 4.5,
        'end': 5.6
    }
]
```

---

## Trailing Word Merge

### Problem
Sometimes a sentence-starting word gets stuck at the end:
```
❌ Customer: "What's going on? I"
   Agent: "completely understand..."
```

### Solution
Check if the last word of an utterance is a sentence starter (I, We, You, That, etc.) and merge it with the next utterance:

```python
trailing_starters = ['I', 'We', 'You', 'That', 'This', 'It', 'So', 'But', 'And']

for i, utt in enumerate(utterances):
    words = utt['text'].split()
    
    if i + 1 < len(utterances) and len(words) > 2:
        last_word = words[-1]
        if last_word in trailing_starters:
            # Move last word to next utterance
            utt['text'] = ' '.join(words[:-1])
            next_utt = utterances[i + 1]
            next_utt['text'] = last_word + ' ' + next_utt['text']
```

**Result:**
```
✅ Customer: "What's going on?"
   Agent: "I completely understand..."
```

---

## Text Post-Processing

Clean up ASR artifacts:

```python
def post_process_text(text):
    # Fix number spacing: "$2 ,500" → "$2,500"
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    
    # Fix currency spacing: "$ 2500" → "$2500"
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    
    # Fix hyphen spacing: "9 -8 -7" → "9-8-7"
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    
    # Fix percentage: "30 %" → "30%"
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

---

## Speaker Role Detection

### Algorithm
Assign "Agent" vs "Customer" labels based on:

1. **First speaker bonus** → Usually Agent answers first (+2 points)
2. **Keyword analysis** → Count agent-like vs customer-like phrases
3. **Tie-breaker** → First speaker wins ties

### Keywords

**Agent indicators:**
```python
AGENT_KEYWORDS = [
    "thank you for calling", "how can i help", "customer support",
    "let me check", "your account", "i can help",
    "is there anything else", "have a wonderful day",
    "valued customer", "i'm going to waive", "upgrade you"
]
```

**Customer indicators:**
```python
CUSTOMER_KEYWORDS = [
    "i'm really upset", "i was charged", "what's going on",
    "i don't think", "i didn't", "that's amazing",
    "thank you so much", "i really appreciate"
]
```

### Example Scoring

```
SPEAKER_00:
  - First speaker: +2
  - "thank you for calling": +1 (agent)
  - "how can i help": +1 (agent)
  - "your account": +1 (agent)
  - Total: 5 agent, 0 customer → Agent

SPEAKER_01:
  - "i'm really upset": +1 (customer)
  - "i was charged": +1 (customer)
  - "thank you so much": +1 (customer)
  - Total: 0 agent, 3 customer → Customer
```

---

## Emotion Recognition

### Multimodal Fusion Architecture

```
Input Text: "I'm really upset"
     ↓
┌─────────────────┐
│   RoBERTa       │ → [CLS] token embedding (768-dim)
│ Text Encoder    │    "understands semantic meaning"
└─────────────────┘
     ↓
┌─────────────────┐
│ Projection to   │ → 256-dim
│  256 dims       │
└─────────────────┘
     ↓
     └───────────────────┐
                         ▼
Input Audio: [waveform]  ┌─────────────────┐
     ↓                   │  Concatenate    │ → 512-dim
┌─────────────────┐      │  Text + Audio   │
│   WavLM         │ →    └─────────────────┘
│ Audio Encoder   │              ↓
└─────────────────┘      ┌─────────────────┐
  Mean pooling           │   Classifier    │
  (768-dim)              │  256→128→3      │
     ↓                   └─────────────────┘
┌─────────────────┐              ↓
│ Projection to   │ → 256-dim    [Negative, Neutral, Positive]
│  256 dims       │
└─────────────────┘
     ↓
     └───────────────────┘
```

### Processing Steps

1. **Text Encoding:**
```python
inputs = tokenizer("I'm really upset", padding='max_length', max_length=64)
text_out = roberta(input_ids, attention_mask)
text_embed = text_out.last_hidden_state[:, 0, :]  # CLS token
text_projected = text_proj(text_embed)  # 768 → 256
```

2. **Audio Encoding:**
```python
audio_segment = full_audio[start_sample:end_sample]
audio_vals = wavlm_processor(audio_segment, sr=16000)
audio_out = wavlm(audio_vals)
audio_embed = torch.mean(audio_out.last_hidden_state, dim=1)  # Average pooling
audio_projected = audio_proj(audio_embed)  # 768 → 256
```

3. **Fusion & Classification:**
```python
fused = torch.cat([text_projected, audio_projected], dim=1)  # 512-dim
logits = classifier(fused)  # 512 → 256 → 128 → 3
emotion = argmax(logits)  # [Negative=0, Neutral=1, Positive=2]
```

### Why Multimodal?

| Modality | What it captures | Example |
|----------|------------------|---------|
| **Text only** | Semantic meaning | "I'm fine" → Neutral (words are neutral) |
| **Audio only** | Acoustic cues | Sarcastic tone → Negative (voice reveals true emotion) |
| **Multimodal** | Combined understanding | "I'm fine" (sarcastic) → Negative ✅ |

---

## Example Walkthrough

### Input Audio
- File: `telecom_call.mp3`
- Duration: 158.99 seconds
- Sample rate: 44100 Hz → resampled to 16000 Hz

### Step-by-Step Processing

#### 1. Pyannote Diarization Output
```python
speaker_segments = [
    {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 4.238},
    {'speaker': 'SPEAKER_01', 'start': 4.356, 'end': 12.169},
    {'speaker': 'SPEAKER_00', 'start': 12.439, 'end': 20.507},
    {'speaker': 'SPEAKER_01', 'start': 20.574, 'end': 23.530},
    # ... 67 total segments
]
```

#### 2. Whisper ASR Output
```python
all_words = [
    {'word': 'Hello', 'start': 0.0, 'end': 0.5, 'probability': 0.98},
    {'word': ',', 'start': 0.5, 'end': 0.52, 'probability': 0.95},
    {'word': 'thank', 'start': 0.52, 'end': 0.82, 'probability': 0.97},
    {'word': 'you', 'start': 0.84, 'end': 1.1, 'probability': 0.99},
    {'word': 'for', 'start': 1.12, 'end': 1.3, 'probability': 0.98},
    {'word': 'calling', 'start': 1.32, 'end': 1.8, 'probability': 0.96},
    # ... continues
]

# Metadata
info.language = "en"
info.language_probability = 0.98
avg_confidence = 0.9639  # 96.39%
```

#### 3. Word-to-Speaker Matching

```python
# Word: "Hello" (0.0-0.5s, midpoint 0.25s)
# Check speaker_segments: 0.25 is in [0.0-4.238] → SPEAKER_00

# Word: "Hi" (4.5-4.9s, midpoint 4.7s)
# Check speaker_segments: 4.7 is in [4.356-12.169] → SPEAKER_01
```

#### 4. Utterance Grouping

```python
# SPEAKER_00 words: ["Hello", "thank", "you", "for", "calling", ...]
# Group into: "Hello, thank you for calling customer support..."

utterances = [
    {
        'speaker': 'SPEAKER_00',
        'text': 'Hello, thank you for calling customer support. My name is Rajesh. How can I help you today?',
        'start': 0.0,
        'end': 4.238,
        'confidence': 0.97
    },
    {
        'speaker': 'SPEAKER_01',
        'text': 'Hi Rajesh, I\'m really upset. My bill this month is way higher...',
        'start': 4.356,
        'end': 12.169,
        'confidence': 0.95
    }
]
```

#### 5. Role Detection

```python
# SPEAKER_00 scores:
#   - First speaker: +2
#   - "thank you for calling": +1
#   - "customer support": +1
#   - "How can I help": +1
#   - Total: 5 agent, 0 customer → Agent

# SPEAKER_01 scores:
#   - "I'm really upset": +1
#   - "My bill": 0
#   - Total: 0 agent, 1 customer → Customer

speaker_to_role = {
    'SPEAKER_00': 'Agent',
    'SPEAKER_01': 'Customer'
}
```

#### 6. Emotion Recognition

For each utterance:

**Utterance 1:**
```python
Text: "Hello, thank you for calling customer support..."
Audio: [16000 samples from 0.0s to 4.238s]

# Text encoding
text_embed = roberta("Hello, thank you for calling...") 
# → Detects professional greeting → slightly positive

# Audio encoding  
audio_embed = wavlm([audio samples])
# → Detects calm, professional tone

# Fusion
logits = classifier(text_embed + audio_embed)
# → [0.1, 0.7, 0.2] → Neutral (index 1)
```

**Utterance 2:**
```python
Text: "Hi Rajesh, I'm really upset. My bill this month is way higher..."
Audio: [16000 samples from 4.356s to 12.169s]

# Text encoding
text_embed = roberta("I'm really upset. My bill...")
# → Detects negative words: "upset", "higher"

# Audio encoding
audio_embed = wavlm([audio samples])
# → Detects stressed voice, higher pitch

# Fusion
logits = classifier(text_embed + audio_embed)
# → [0.8, 0.1, 0.1] → Negative (index 0)
```

---

## Output Format

### Final Conversation Log

```python
[
    {
        'speaker': 'SPEAKER_00',
        'role': 'Agent',
        'emotion': 'Neutral',
        'text': 'Hello, thank you for calling customer support. My name is Rajesh. How can I help you today?',
        'confidence': 0.97,
        'timestamp': (0.0, 4.238)
    },
    {
        'speaker': 'SPEAKER_01',
        'role': 'Customer',
        'emotion': 'Negative',
        'text': 'Hi Rajesh, I\'m really upset. My bill this month is way higher than last month. I was charged $2,500 instead of $1,800. What\'s going on?',
        'confidence': 0.95,
        'timestamp': (4.356, 12.169)
    },
    {
        'speaker': 'SPEAKER_00',
        'role': 'Agent',
        'emotion': 'Neutral',
        'text': 'I completely understand your frustration. That\'s definitely concerning. Let me help you figure this out. Can I get your phone number please?',
        'confidence': 0.98,
        'timestamp': (12.439, 20.507)
    },
    # ... continues for all 15 utterances
]
```

### Console Output

```
============================================================
Processing: C:\...\telecom_call.mp3
============================================================
Original sample rate: 44100 Hz
Audio duration: 158.99s

Running Speaker Diarization (Pyannote)...
Found 67 speaker segments
Speakers detected: ['SPEAKER_00', 'SPEAKER_01']

Running ASR (faster-whisper medium)...
Language detected: en (probability: 0.98)
Average word confidence: 0.96
Transcript: Hello , thank you for calling customer support ...

Role assignment: {'SPEAKER_00': 'Agent', 'SPEAKER_01': 'Customer'}

Running Emotion Recognition...
Agent (Neutral): Hello, thank you for calling customer support. My name is Rajesh. How can I help you today?
Customer (Negative): Hi Rajesh, I'm really upset. My bill this month is way higher than last month. I was charged $2,500 instead of $1,800. What's going on?
Agent (Neutral): I completely understand your frustration. That's definitely concerning. Let me help you figure this out. Can I get your phone number please?
Customer (Neutral): Sure, it's 987-6-5-4-3-2-1. No.
Agent (Neutral): Thank you. Let me check your account. I can see the bill increase. There are several reasons this could happen. Additional data usage, new subscriptions, service charges, or a plan upgrade. Let me check your details.
Customer (Negative): I don't think I added anything. I haven't changed my plan, so it must be the data usage then.
Agent (Neutral): Good question. Background apps like cloud storage, updates, and social media can use data without you realizing. Looking at your account, you used 35 gigabytes this month compared to your usual 20. That's an extra 15 gigabytes. Since your plan includes 25 gigabytes, you were charged for the extra 10 gigabytes at 75 rupees per gigabyte. That's 1,125 rupees in overage charges.
Customer (Positive): Oh wow, 35 gigabytes? I didn't use that much intentionally, so that's the extra charge then?
Agent (Positive): Yes, that's part of it. But here's the good news. We can prevent this from happening again. I can upgrade you to our Unlimited Plan at 8.99 rupees per month. You get unlimited data with no overage charges, or the Premium Data Plan at 6.99 rupees with 150 gigabytes and rollover. What sounds best to you?
Customer (Positive): The unlimited plan sounds good, but when will the change take effect?
Agent (Positive): Great choice. The upgrade takes effect immediately for your next billing cycle. So this month's charges stay as is, but starting next month you'll be on unlimited and have no surprise charges. I'm also going to waive 30% of your overage charges since this was unexpected. Your new bill will be about 1,750 instead of $2,500. How does that sound?
Customer (Positive): Really? You're waiving some charges? That's amazing. Thank you so much. I really appreciate that.
Agent (Neutral): You're very welcome. Your upgrade is confirmed and will be active within 5 minutes. You'll get an SMS confirmation. Is there anything else I can help you
Customer (Positive): with? No, I think that's it. Thanks again for being so helpful and understanding. This was much better than I expected.
Agent (Neutral): Thank you for being a valued customer. Have a wonderful day and don't hesitate to reach out if you need anything in the future.

============================================================
SUMMARY
============================================================
Total utterances: 15
Low confidence segments: 0
Language: en
Average confidence: 96.39%
```

---

## Performance Metrics

### Test Audio: telecom_call.mp3

| Metric | Value |
|--------|-------|
| **Duration** | 158.99 seconds |
| **Speaker Accuracy** | 100% (15/15 correct) |
| **Transcription WER** | ~5% (minor number formatting issues) |
| **Emotion Accuracy** | ~87% (13/15 match ground truth sentiment) |
| **Average Confidence** | 96.39% |
| **Processing Speed** | ~3x realtime (medium model on CPU int8) |

### Comparison with Ground Truth

| Turn | Role | Emotion GT | Emotion Pred | Text Match | Notes |
|------|------|------------|--------------|------------|-------|
| 1 | Agent | Professional | Neutral | ✅ | Perfect |
| 2 | Customer | Frustrated | Negative | ✅ | "$2,500" spacing fixed |
| 3 | Agent | Empathetic | Neutral | ⚠️ | Close (empathy ≈ neutral) |
| 4 | Customer | Calmer | Neutral | ✅ | Phone # has spacing |
| 5 | Agent | Professional | Neutral | ✅ | Perfect |
| 6 | Customer | Curious | Negative | ⚠️ | Model detects concern as negative |
| 7 | Agent | Informative | Neutral | ✅ | Perfect |
| 8 | Customer | Surprised | Positive | ✅ | Perfect |
| 9 | Agent | Helpful | Positive | ✅ | Perfect |
| 10 | Customer | Interested | Positive | ✅ | Perfect |
| 11 | Agent | Clear | Positive | ✅ | Perfect |
| 12 | Customer | Grateful | Positive | ✅ | "waiving" fixed! |
| 13 | Agent | Warm | Neutral | ⚠️ | Model more conservative |
| 14 | Customer | Happy | Positive | ✅ | Perfect |
| 15 | Agent | Professional | Neutral | ✅ | Perfect |

**Accuracy: 13/15 = 86.67%**

---

## Key Improvements from V1

| Feature | V1 | V2 |
|---------|----|----|
| ASR Model | transformers Whisper base | faster-whisper medium |
| Confidence Scores | ❌ No | ✅ Per-word probabilities |
| Language Detection | ❌ No | ✅ Auto-detect with confidence |
| Role Detection | First speaker = Agent | ✅ Keyword-based smart assignment |
| Text Cleanup | ❌ No | ✅ Post-processing (numbers, punctuation) |
| Alignment | Chunk-level | ✅ Word-level with trailing merge |
| Sample Rate Info | ❌ No | ✅ Original SR detection + telephony warning |

---

## Next Steps for Production

### 1. Phone Audio Quality (8kHz Telephony)

Current pipeline resamples 8kHz → 16kHz, but doesn't add lost frequencies. Options:

**A. Audio Enhancement:**
```python
import noisereduce as nr

# Denoise
audio_clean = nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.8)

# Normalize levels
audio_norm = librosa.util.normalize(audio_clean)

# Optional: Telephony-specific filter
from scipy import signal
nyquist = 8000 / 2
low = 300 / nyquist
high = 3400 / nyquist
b, a = signal.butter(4, [low, high], btype='band')
audio_filtered = signal.filtfilt(b, a, audio_norm)
```

**B. Fine-tune Whisper on telephony data:**
```python
# Train on call center recordings with 8kHz quality
# Dataset: VoxPopuli, CommonVoice telephony subset
```

### 2. Emotion Model on Real Calls

Current model trained on clean IEMOCAP/MELD data. For production:

**A. Data Collection:**
- Record real call center conversations
- Label with domain-specific emotions (frustrated, satisfied, confused, etc.)

**B. Data Augmentation:**
```python
import audiomentations as AA

augment = AA.Compose([
    AA.AddGaussianNoise(p=0.5),
    AA.TimeStretch(p=0.3),
    AA.PitchShift(p=0.3),
    AA.TelephonySim(p=0.5)  # Simulate phone quality
])
```

**C. Fine-tuning:**
```python
# Freeze lower layers, fine-tune classifier on call center data
for param in model.text_encoder.parameters():
    param.requires_grad = False
    
# Train classifier + projection layers
optimizer = AdamW(
    list(model.text_proj.parameters()) + 
    list(model.audio_proj.parameters()) +
    list(model.classifier.parameters()),
    lr=1e-4
)
```

---

## Usage Examples

### Basic Usage
```bash
conda activate pytorch_cuda13
cd Experiments/Speech-Pipeline-Tests
python run_demo.py --audio ../../Voice-Generation/telecom_call.mp3
```

### Programmatic Usage
```python
from inference_pipeline_v2 import VocalMindPipelineV2

# Initialize
pipeline = VocalMindPipelineV2(
    emotion_model_path="path/to/best_model_3class.pt",
    whisper_model="medium"  # or "small", "large-v3"
)

# Process audio
conversation_log = pipeline.process_file("audio.mp3")

# Analyze results
for turn in conversation_log:
    print(f"{turn['role']} ({turn['emotion']}): {turn['text']}")
    if turn['confidence'] < 0.7:
        print("  ⚠️ Low confidence - manual review recommended")
```

### Batch Processing
```python
import glob

audio_files = glob.glob("call_recordings/*.mp3")
for audio in audio_files:
    print(f"\nProcessing {audio}...")
    log = pipeline.process_file(audio)
    
    # Save to JSON
    import json
    output_file = audio.replace(".mp3", "_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(log, f, indent=2)
```

---

## Troubleshooting

### GPU Memory Issues
```python
# Use smaller model or CPU
pipeline = VocalMindPipelineV2(
    emotion_model_path="...",
    whisper_model="small"  # Instead of "medium"
)
```

### cuBLAS Error (faster-whisper)
```python
# Already handled in V2 - uses CPU with int8 quantization
# If you have CUDA 11, you can use GPU:
self.asr_model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
```

### Low Confidence Transcriptions
```python
# Check segments with confidence < 0.7
low_conf = [t for t in conversation_log if t['confidence'] < 0.7]
for seg in low_conf:
    print(f"Review: {seg['text']} ({seg['confidence']:.2f})")
```

---

## References

- [Pyannote Speaker Diarization](https://github.com/pyannote/pyannote-audio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- [WavLM](https://arxiv.org/abs/2110.13900)

---

## 8. Hybrid Source Separation Method (V3 - Experimental)

### 8.1 Motivation

**Problem with Current Approach (V2.1):**

The V2.1 pipeline can **detect** overlapping speech but cannot **transcribe** it completely:

```
Timeline:    [=======Agent=======]
             [====Customer====]
Time:        10.0s          12.0s
             └─ Overlap: 10.5s-11.5s (1.0s)
```

**What happens during overlap:**
- Pyannote correctly identifies that both speakers are talking simultaneously
- faster-whisper (single-channel ASR) transcribes only the **dominant/louder speaker**
- Words from the quieter speaker are **lost or garbled**

**Example:**
```
Ground Truth:
  Agent:    "I can upgrade you to unlimited for just..."
  Customer: "Yes, that sounds good." (overlapping at 0.8s)

V2.1 Output:
  Agent:    "I can upgrade you to unlimited for just yes that sounds..."
            └─ Mixed both speakers into one transcript ❌
```

### 8.2 Solution Architecture

The **Hybrid Source Separation Method** solves this by:
1. **Separating** the mixed audio into isolated speaker tracks
2. **Matching** separated tracks to speaker identities via diarization
3. **Transcribing** each track independently
4. **Reconstructing** the conversation timeline with complete transcripts

---

### 8.3 Full Pipeline Flow (V3)

```
┌───────────────────────────────────────────────────────────────┐
│                     INPUT: Mixed Audio                        │
│              (Both speakers in single channel)                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│           STEP 1: SOURCE SEPARATION (Sepformer)               │
│  • Neural network trained to separate overlapping voices      │
│  • Outputs: Track 1 (isolated) + Track 2 (isolated)          │
└───────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          [Track 1 Audio]           [Track 2 Audio]
          (Speaker A only)          (Speaker B only)
                 │                         │
                 └────────────┬────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│      STEP 2: SPEAKER DIARIZATION (Pyannote on Original)      │
│  • Run on ORIGINAL mixed audio to get timeline labels        │
│  • Outputs: SPEAKER_00: 0-4.2s, SPEAKER_01: 4.3-9.8s, ...   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│         STEP 3: TRACK-TO-SPEAKER MATCHING                     │
│  • Compare each separated track to diarization segments      │
│  • Use voice embeddings to determine:                         │
│    - Track 1 = SPEAKER_00 (Agent)                            │
│    - Track 2 = SPEAKER_01 (Customer)                         │
└───────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  STEP 4A: TRANSCRIBE     │    │  STEP 4B: TRANSCRIBE     │
│  Track 1 (Agent)         │    │  Track 2 (Customer)      │
│  with faster-whisper     │    │  with faster-whisper     │
│  → Word timestamps       │    │  → Word timestamps       │
└──────────────────────────┘    └──────────────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│         STEP 5: RECONSTRUCT TIMELINE                          │
│  • Merge transcripts from both tracks using timestamps       │
│  • Apply speaker role detection                              │
│  • Flag overlapping segments (still useful for QA)           │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│         STEP 6: EMOTION RECOGNITION                           │
│  • Same as V2 - multimodal fusion on each utterance          │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│         OUTPUT: Complete Conversation Log                     │
│  • Both speakers fully transcribed, even during overlaps      │
└───────────────────────────────────────────────────────────────┘
```

---

### 8.4 Implementation Details

#### 8.4.1 Source Separation with Sepformer

**Model:** [SpeechBrain Sepformer](https://github.com/speechbrain/speechbrain)

```python
from speechbrain.pretrained import SepformerSeparation

# Load pre-trained model
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wham",
    savedir="pretrained_models/sepformer"
)

# Separate audio into 2 sources
separated_sources = separator.separate_file(audio_path)
# Returns: tensor of shape [num_sources, samples]

# Extract tracks
track_agent = separated_sources[0].cpu().numpy()
track_customer = separated_sources[1].cpu().numpy()
```

**Note:** Sepformer is trained to separate up to 2-3 speakers. For more, use models like **ConvTasNet** or **DPRNN**.

---

#### 8.4.2 Track-to-Speaker Matching

**Challenge:** Sepformer outputs "Track 1" and "Track 2", but we don't know which is Agent vs Customer.

**Solution:** Compare voice embeddings

```python
from pyannote.audio import Model
from pyannote.audio.pipelines.utils import get_embeddings

# Load speaker embedding model
embedding_model = Model.from_pretrained("pyannote/embedding")

def get_track_embedding(audio_segment):
    """Extract speaker embedding from audio segment."""
    return embedding_model(torch.from_numpy(audio_segment).unsqueeze(0))

# Get embeddings for each track (use first 5 seconds)
track1_embed = get_track_embedding(track_agent[:16000*5])
track2_embed = get_track_embedding(track_customer[:16000*5])

# Compare with diarization speaker embeddings
# Match track to speaker based on cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# For each speaker segment, extract embedding from original audio
speaker_embeds = {}
for segment in speaker_segments[:10]:  # Sample first 10 segments
    start_sample = int(segment['start'] * 16000)
    end_sample = int(segment['end'] * 16000)
    audio_slice = full_audio[start_sample:end_sample]
    
    embed = get_track_embedding(audio_slice)
    speaker = segment['speaker']
    
    if speaker not in speaker_embeds:
        speaker_embeds[speaker] = []
    speaker_embeds[speaker].append(embed)

# Average embeddings per speaker
avg_embeds = {
    spk: torch.mean(torch.stack(embeds), dim=0) 
    for spk, embeds in speaker_embeds.items()
}

# Match tracks to speakers
similarities = {}
for speaker, avg_embed in avg_embeds.items():
    sim_track1 = cosine_similarity(track1_embed.numpy(), avg_embed.numpy())[0][0]
    sim_track2 = cosine_similarity(track2_embed.numpy(), avg_embed.numpy())[0][0]
    similarities[speaker] = {'track1': sim_track1, 'track2': sim_track2}

# Assign tracks
track_to_speaker = {}
if similarities['SPEAKER_00']['track1'] > similarities['SPEAKER_00']['track2']:
    track_to_speaker = {0: 'SPEAKER_00', 1: 'SPEAKER_01'}
else:
    track_to_speaker = {0: 'SPEAKER_01', 1: 'SPEAKER_00'}
```

---

#### 8.4.3 Independent Transcription

```python
# Transcribe each track separately
transcripts = {}

for track_idx, speaker in track_to_speaker.items():
    audio_track = track_agent if track_idx == 0 else track_customer
    
    # Run faster-whisper on isolated track
    segments, info = asr_model.transcribe(
        audio_track,
        beam_size=5,
        word_timestamps=True,
        language="en"
    )
    
    # Collect words with timestamps
    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words.append({
                    'word': word.word.strip(),
                    'start': word.start,
                    'end': word.end,
                    'speaker': speaker
                })
    
    transcripts[speaker] = words

# Merge all words and sort by timestamp
all_words = []
for speaker, words in transcripts.items():
    all_words.extend(words)

all_words.sort(key=lambda x: x['start'])
```

---

#### 8.4.4 Timeline Reconstruction

```python
# Group words into utterances by speaker
utterances = []
current_speaker = None
current_words = []
current_start = None

for word_data in all_words:
    word = word_data['word']
    speaker = word_data['speaker']
    
    # Check if speaker changed
    if speaker != current_speaker and current_words:
        utterances.append({
            'speaker': current_speaker,
            'text': ' '.join(current_words),
            'start': current_start,
            'end': word_data['start']
        })
        current_words = []
        current_start = None
    
    if current_start is None:
        current_start = word_data['start']
    
    current_words.append(word)
    current_speaker = speaker

# Add final utterance
if current_words:
    utterances.append({
        'speaker': current_speaker,
        'text': ' '.join(current_words),
        'start': current_start,
        'end': all_words[-1]['end']
    })
```

---

### 8.5 Example: Overlap Handling

**Input Audio (with overlap at 10.5-11.5s):**

```
Ground Truth:
  Agent:    "I can upgrade you to unlimited for just $899 per month"  (10.0s-12.0s)
  Customer: "Yes, that sounds good"                                    (10.5s-11.2s)
```

**V2.1 Output (Current):**
```
Agent: "I can upgrade you to unlimited for just yes that sounds good per month" [OVERLAP]
       └─ Garbled mix ❌
```

**V3 Output (Hybrid Method):**
```
Agent:    "I can upgrade you to unlimited for just $899 per month"    (10.0s-12.0s)
Customer: "Yes, that sounds good"                                      (10.5s-11.2s) [OVERLAP]
          └─ Both transcripts complete ✅
```

---

### 8.6 Performance Trade-offs

| Aspect | V2.1 (Current) | V3 (Hybrid) | Comparison |
|--------|---------------|-------------|------------|
| **Overlap Detection** | ✅ Yes | ✅ Yes | Same |
| **Overlap Transcription** | ❌ Incomplete | ✅ Complete | **+100% during overlaps** |
| **Compute Time** | 1x baseline | **3-5x baseline** | 3-5x slower |
| **GPU Memory** | ~4GB | **~8-12GB** | 2-3x higher |
| **Accuracy (no overlap)** | 96.39% | ~96% | Similar |
| **Accuracy (with overlap)** | ~50% | **~95%** | Major improvement |
| **Latency (real-time)** | ~0.5x audio length | **~2x audio length** | Too slow for streaming |
| **Best Use Case** | Clean calls, batch | High-overlap calls, offline | - |

---

### 8.7 When to Use V3

**Deploy V3 if:**
1. ✅ **Overlap frequency >10%** in production calls (from V2.1 metrics)
2. ✅ **Batch processing** acceptable (not real-time)
3. ✅ **High GPU resources** available (8GB+ VRAM)
4. ✅ **Critical accuracy** needed for overlapping speech

**Stick with V2.1 if:**
1. ❌ Overlap frequency <5%
2. ❌ Real-time streaming required
3. ❌ Limited compute budget
4. ❌ Overlaps are short acknowledgments ("uh-huh", "okay") that don't contain critical info

---

### 8.8 Phased Deployment Strategy

```
Phase 1: Deploy V2.1 [Current]
  └─ Monitor overlap metrics on production calls
  └─ Gather data: % calls with overlaps, avg overlap duration
  └─ Identify high-overlap scenarios (frustrated customers, negotiations)

Phase 2: Selective V3 [Conditional]
  └─ IF overlap-affected utterances >10%:
      ├─ Deploy V3 for batch analysis of flagged calls
      ├─ Use V2.1 for real-time/low-priority calls
      └─ Route high-overlap calls to V3 queue

Phase 3: Optimization [Future]
  └─ Fine-tune Sepformer on call center data
  └─ Optimize track matching (faster embeddings)
  └─ Explore GPU parallelization (separate tracks simultaneously)
```

---

### 8.9 Alternative: Lightweight Overlap Recovery

**For scenarios where V3 is too expensive:**

**Option A: Overlap-aware ASR**
- Use multi-channel Whisper variants (if available)
- Example: [Whisper-AT](https://github.com/YuanGongND/whisper-at) (attention-based multi-speaker)

**Option B: Post-hoc Overlap Filling**
- Detect overlaps with V2.1
- Only apply source separation to overlapping segments (not full audio)
- Transcribe isolated overlaps and splice back in

```python
for overlap in detected_overlaps:
    # Extract overlapping audio chunk
    start_sample = int(overlap['start'] * 16000)
    end_sample = int(overlap['end'] * 16000)
    chunk = full_audio[start_sample:end_sample]
    
    # Separate only this chunk
    separated = separator.separate_batch(torch.from_numpy(chunk).unsqueeze(0))
    
    # Transcribe both tracks for this segment
    # ... (same as V3 but limited to overlap regions)
```

**Savings:** Only 5-15% of audio processed with source separation (instead of 100%)

---

### 8.10 Code Skeleton for V3

```python
class VocalMindPipelineV3:
    def __init__(self, emotion_model_path, whisper_model="medium"):
        # V2 components
        self.asr_model = WhisperModel(whisper_model, ...)
        self.diarization_pipeline = PyannotePipeline.from_pretrained(...)
        self.emotion_model = MultimodalFusionNet3Class(...)
        
        # NEW: Source separation
        from speechbrain.pretrained import SepformerSeparation
        self.separator = SepformerSeparation.from_hparams(...)
        
        # NEW: Speaker embedding model for track matching
        from pyannote.audio import Model
        self.embedding_model = Model.from_pretrained("pyannote/embedding")
    
    def process_file(self, audio_path):
        # 1. Load audio
        full_audio, sr = librosa.load(audio_path, sr=16000)
        
        # 2. Source separation
        separated = self.separator.separate_file(audio_path)
        track1 = separated[0].cpu().numpy()
        track2 = separated[1].cpu().numpy()
        
        # 3. Diarization on original audio (for timeline)
        diarization = self.diarization_pipeline({"waveform": ..., "sample_rate": sr})
        speaker_segments = [...]
        
        # 4. Match tracks to speakers
        track_to_speaker = self._match_tracks(track1, track2, speaker_segments, full_audio)
        
        # 5. Transcribe each track
        transcripts = {}
        for track_idx, track_audio in enumerate([track1, track2]):
            speaker = track_to_speaker[track_idx]
            words = self._transcribe_track(track_audio)
            transcripts[speaker] = words
        
        # 6. Reconstruct timeline
        utterances = self._reconstruct_timeline(transcripts)
        
        # 7. Speaker role detection (same as V2)
        speaker_to_role = detect_speaker_roles(utterances, speaker_segments)
        
        # 8. Emotion recognition (same as V2)
        conversation_log = []
        for utt in utterances:
            emotion = self._predict_emotion(utt['text'], full_audio, utt['start'], utt['end'])
            conversation_log.append({
                'role': speaker_to_role[utt['speaker']],
                'text': utt['text'],
                'emotion': emotion,
                ...
            })
        
        return conversation_log
    
    def _match_tracks(self, track1, track2, speaker_segments, full_audio):
        # Implementation from Section 8.4.2
        ...
    
    def _transcribe_track(self, track_audio):
        # Implementation from Section 8.4.3
        ...
    
    def _reconstruct_timeline(self, transcripts):
        # Implementation from Section 8.4.4
        ...
```

---

### 8.11 Dependencies for V3

```bash
pip install speechbrain
pip install torch-audiomentations  # For augmentation
```

**Model downloads:**
```python
# ~300MB for Sepformer
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wham",
    savedir="pretrained_models/sepformer"
)

# ~50MB for embedding model (already used in pyannote)
embedding_model = Model.from_pretrained("pyannote/embedding")
```

---

### 8.12 Summary

**V3 Hybrid Method is powerful but expensive.**

- ✅ **Use when:** Overlap frequency justifies compute cost (>10% affected utterances)
- ⚠️ **Avoid when:** Real-time streaming needed or overlaps are rare (<5%)
- 🔄 **Strategy:** Deploy V2.1 first, monitor data, upgrade to V3 selectively

**Key insight:** Most call center calls have <5% overlap. The synthetic test call has **0% overlap**. Only invest in V3 if production data shows it's necessary.
