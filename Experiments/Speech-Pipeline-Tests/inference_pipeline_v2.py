"""
VocalMind Inference Pipeline v2 - Production-Ready
===================================================
Upgrades from v1:
- faster-whisper with large-v3 model (better accuracy)
- Confidence scores for transcriptions
- Language detection
- Smarter speaker role detection (Agent vs Customer)
- Text post-processing (fix number spacing)
- Original sample rate detection
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
import warnings
import re
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Suppress warnings
warnings.filterwarnings("ignore")

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# 1. EMOTION MODEL CONFIGURATION & ARCHITECTURE
# -----------------------------------------------------------------------------
class EmotionConfig:
    TEXT_MODEL = "roberta-base"
    AUDIO_MODEL = "microsoft/wavlm-base-plus"
    NUM_CLASSES = 3
    HIDDEN_DIM = 768
    PROJECTED_DIM = 256
    DROPOUT = 0.3
    EMOTION_LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

class MultimodalFusionNet3Class(nn.Module):
    def __init__(self, config=EmotionConfig):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL)
        self.audio_encoder = AutoModel.from_pretrained(config.AUDIO_MODEL)
        
        self.text_proj = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.PROJECTED_DIM), 
            nn.GELU(), 
            nn.Dropout(config.DROPOUT)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.PROJECTED_DIM), 
            nn.GELU(), 
            nn.Dropout(config.DROPOUT)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.PROJECTED_DIM * 2, 256), 
            nn.GELU(), 
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 128), 
            nn.GELU(), 
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, audio_values):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_out.last_hidden_state[:, 0, :]
        
        audio_out = self.audio_encoder(input_values=audio_values)
        audio_embed = torch.mean(audio_out.last_hidden_state, dim=1)
        
        fused = torch.cat([self.text_proj(text_embed), self.audio_proj(audio_embed)], dim=1)
        return self.classifier(fused)

# -----------------------------------------------------------------------------
# 2. TEXT POST-PROCESSING
# -----------------------------------------------------------------------------
def post_process_text(text):
    """Clean up ASR output: fix spacing around numbers and punctuation."""
    # Fix number spacing: "$2 ,500" -> "$2,500"
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    # Fix spacing after currency: "$ 2500" -> "$2500"
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    # Fix spacing around hyphens in numbers: "9 -8 -7" -> "9-8-7"  
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    # Fix percentage spacing: "30 %" -> "30%"
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------------------------------------------------------
# 3. SPEAKER ROLE DETECTION
# -----------------------------------------------------------------------------
# Keywords that typically indicate Agent speech
AGENT_KEYWORDS = [
    "thank you for calling", "how can i help", "customer support",
    "let me check", "your account", "i can help", "is there anything else",
    "have a wonderful day", "valued customer", "i'm going to waive",
    "upgrade you", "your bill", "your plan"
]

CUSTOMER_KEYWORDS = [
    "i'm really upset", "i was charged", "what's going on",
    "i don't think", "i didn't", "that's amazing", "thank you so much",
    "i really appreciate", "much better than i expected"
]

def detect_speaker_roles(utterances, speaker_segments):
    """
    Detect which speaker is Agent vs Customer based on:
    1. Who speaks first (usually Agent in call center)
    2. Keyword analysis
    3. Speaking patterns
    """
    unique_speakers = list(dict.fromkeys([s['speaker'] for s in speaker_segments]))
    if len(unique_speakers) < 2:
        return {unique_speakers[0]: "Agent"} if unique_speakers else {}
    
    # Score each speaker
    speaker_scores = {spk: {"agent": 0, "customer": 0} for spk in unique_speakers}
    
    # First speaker bonus (Agent usually answers first)
    first_speaker = speaker_segments[0]['speaker'] if speaker_segments else None
    if first_speaker:
        speaker_scores[first_speaker]["agent"] += 2
    
    # Analyze utterance content
    for utt in utterances:
        text_lower = utt['text'].lower()
        speaker = utt['speaker']
        
        for kw in AGENT_KEYWORDS:
            if kw in text_lower:
                speaker_scores[speaker]["agent"] += 1
        
        for kw in CUSTOMER_KEYWORDS:
            if kw in text_lower:
                speaker_scores[speaker]["customer"] += 1
    
    # Assign roles based on scores
    speaker_to_role = {}
    for spk in unique_speakers:
        scores = speaker_scores[spk]
        if scores["agent"] > scores["customer"]:
            speaker_to_role[spk] = "Agent"
        elif scores["customer"] > scores["agent"]:
            speaker_to_role[spk] = "Customer"
        else:
            # Tie-breaker: first speaker is Agent
            speaker_to_role[spk] = "Agent" if spk == first_speaker else "Customer"
    
    # Ensure we have both roles
    roles_assigned = set(speaker_to_role.values())
    if "Customer" not in roles_assigned:
        # Find speaker with lowest agent score
        non_agent = min(unique_speakers, key=lambda s: speaker_scores[s]["agent"])
        speaker_to_role[non_agent] = "Customer"
    
    return speaker_to_role

# -----------------------------------------------------------------------------
# 4. PIPELINE CLASS (faster-whisper large-v3 + Pyannote)
# -----------------------------------------------------------------------------
class VocalMindPipelineV2:
    def __init__(self, emotion_model_path, whisper_model="medium"):
        """
        Production pipeline using:
        - faster-whisper medium for ASR (good speed/accuracy balance)
        - Pyannote 3.1 for speaker diarization
        - Multimodal emotion recognition
        
        Options for whisper_model: "base", "small", "medium", "large-v3"
        """
        # --- Load Emotion Model ---
        print(f"Loading Emotion Model from {emotion_model_path}...")
        self.emotion_model = MultimodalFusionNet3Class(EmotionConfig).to(DEVICE)
        
        if not os.path.exists(emotion_model_path):
            raise FileNotFoundError(f"Emotion model not found at {emotion_model_path}")
            
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        self.emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=map_location, weights_only=True))
        self.emotion_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(EmotionConfig.TEXT_MODEL)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(EmotionConfig.AUDIO_MODEL)
        print("Emotion Model Loaded.")

        # --- Load faster-whisper large-v3 ---
        print(f"Loading faster-whisper ASR ({whisper_model})...")
        from faster_whisper import WhisperModel
        
        # Use CUDA with float16 for speed, fallback to CPU int8 if no GPU
        if torch.cuda.is_available():
            self.asr_model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
        else:
            self.asr_model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print(f"faster-whisper {whisper_model} Loaded.")
        
        # --- Load Pyannote for Speaker Diarization ---
        print("Loading Pyannote Speaker Diarization model...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Please add your HuggingFace token.")
        
        from pyannote.audio import Pipeline as PyannotePipeline
        self.diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        print("Pyannote Diarization Loaded.")
    
    def process_file(self, audio_path):
        """Runs the full pipeline on a single audio file."""
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return []

        print(f"\n{'='*60}")
        print(f"Processing: {audio_path}")
        print(f"{'='*60}")
        
        # Check original sample rate
        original_sr = librosa.get_samplerate(audio_path)
        print(f"Original sample rate: {original_sr} Hz")
        if original_sr <= 8000:
            print("⚠️  Telephony audio detected (≤8kHz) - quality may be limited")
        
        # Load audio at 16kHz
        print("Loading audio...")
        full_audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(full_audio) / sr
        print(f"Audio duration: {duration:.2f}s")
        
        # --- Speaker Diarization ---
        print("\nRunning Speaker Diarization (Pyannote)...")
        waveform_tensor = torch.from_numpy(full_audio).unsqueeze(0)
        audio_input = {"waveform": waveform_tensor, "sample_rate": sr}
        diarization_result = self.diarization_pipeline(audio_input)
        
        diarization = diarization_result.speaker_diarization
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        
        print(f"Found {len(speaker_segments)} speaker segments")
        unique_speakers = list(dict.fromkeys([s['speaker'] for s in speaker_segments]))
        print(f"Speakers detected: {unique_speakers}")
        
        # --- ASR with faster-whisper ---
        print("\nRunning ASR (faster-whisper large-v3)...")
        segments, info = self.asr_model.transcribe(
            full_audio,
            beam_size=5,
            word_timestamps=True,
            language=None  # Auto-detect
        )
        
        # Collect all words with timestamps and confidence
        all_words = []
        total_confidence = 0
        word_count = 0
        
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })
                    total_confidence += word.probability
                    word_count += 1
        
        avg_confidence = total_confidence / word_count if word_count > 0 else 0
        print(f"Language detected: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Average word confidence: {avg_confidence:.2f}")
        print(f"Transcript: {' '.join([w['word'] for w in all_words[:20]])}..." if len(all_words) > 20 else f"Transcript: {' '.join([w['word'] for w in all_words])}")
        
        # --- Match words to speakers ---
        def get_speaker_at_time(t):
            for seg in speaker_segments:
                if seg['start'] <= t <= seg['end']:
                    return seg['speaker']
            min_dist = float('inf')
            closest = unique_speakers[0] if unique_speakers else None
            for seg in speaker_segments:
                seg_mid = (seg['start'] + seg['end']) / 2
                dist = abs(t - seg_mid)
                if dist < min_dist:
                    min_dist = dist
                    closest = seg['speaker']
            return closest
        
        # Group words by speaker
        utterances = []
        current_speaker = None
        current_words = []
        current_start = None
        current_end = None
        current_confidences = []
        
        for w in all_words:
            word = w['word']
            word_start = w['start']
            word_end = w['end']
            word_prob = w['probability']
            
            if not word:
                continue
            
            word_mid = (word_start + word_end) / 2
            word_speaker = get_speaker_at_time(word_mid)
            
            if word_speaker != current_speaker and current_words:
                utterances.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_words),
                    'start': current_start,
                    'end': current_end,
                    'confidence': np.mean(current_confidences) if current_confidences else 0
                })
                current_words = []
                current_confidences = []
                current_start = None
            
            if current_start is None:
                current_start = word_start
            current_words.append(word)
            current_confidences.append(word_prob)
            current_end = word_end
            current_speaker = word_speaker
        
        if current_words:
            utterances.append({
                'speaker': current_speaker,
                'text': ' '.join(current_words),
                'start': current_start,
                'end': current_end,
                'confidence': np.mean(current_confidences) if current_confidences else 0
            })
        
        # --- Post-process: merge trailing short segments ---
        final_utterances = []
        trailing_starters = ['I', 'We', 'You', 'That', 'This', 'It', 'So', 'But', 'And', 'Or', 'If', 'Yes', 'No', 'Ok', 'Okay']
        
        for i, utt in enumerate(utterances):
            utt = utt.copy()
            words = utt['text'].split()
            
            if i + 1 < len(utterances) and len(words) > 2:
                next_utt = utterances[i + 1]
                last_word = words[-1]
                if last_word in trailing_starters:
                    utt['text'] = ' '.join(words[:-1])
                    utt['end'] = utt['end'] - 0.3
                    next_utt['text'] = last_word + ' ' + next_utt['text']
            
            # Apply text post-processing
            utt['text'] = post_process_text(utt['text'])
            
            if utt['text'].strip():
                final_utterances.append(utt)
        
        utterances = final_utterances
        
        # --- Smart speaker role detection ---
        speaker_to_role = detect_speaker_roles(utterances, speaker_segments)
        print(f"Role assignment: {speaker_to_role}")
        
        # --- Emotion Recognition ---
        print("\nRunning Emotion Recognition...")
        conversation_log = []
        low_confidence_count = 0
        
        for utt in utterances:
            text = utt['text']
            start_sec = utt['start']
            end_sec = utt['end']
            speaker = utt['speaker']
            confidence = utt.get('confidence', 1.0)
            
            if not text.strip():
                continue
            
            # Flag low confidence
            confidence_flag = ""
            if confidence < 0.7:
                confidence_flag = " [LOW CONF]"
                low_confidence_count += 1
            
            # Slice audio
            start_sample = int(start_sec * 16000)
            end_sample = int(end_sec * 16000)
            audio_segment = full_audio[start_sample:end_sample]
            
            if len(audio_segment) < 1600:
                continue
            
            # Text preprocessing
            inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
            
            # Audio feature extraction
            MAX_AUDIO_LEN = 16000 * 6
            if len(audio_segment) < MAX_AUDIO_LEN:
                pad_len = MAX_AUDIO_LEN - len(audio_segment)
                audio_segment = np.concatenate([audio_segment, np.zeros(pad_len)])
            else:
                audio_segment = audio_segment[:MAX_AUDIO_LEN]
            
            audio_vals = self.audio_processor(audio_segment, sampling_rate=16000, return_tensors="pt").input_values
            
            # Inference
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            audio_vals = audio_vals.to(DEVICE)
            
            with torch.no_grad():
                logits = self.emotion_model(inputs['input_ids'], inputs['attention_mask'], audio_vals)
                pred_idx = torch.argmax(logits, dim=1).item()
                emotion = EmotionConfig.EMOTION_LABELS[pred_idx]
            
            role = speaker_to_role.get(speaker, "Unknown")
            
            log_line = f"{role} ({emotion}): {text}{confidence_flag}"
            print(log_line)
            
            conversation_log.append({
                'speaker': speaker,
                'role': role,
                'emotion': emotion,
                'text': text,
                'confidence': confidence,
                'timestamp': (start_sec, end_sec)
            })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total utterances: {len(conversation_log)}")
        print(f"Low confidence segments: {low_confidence_count}")
        print(f"Language: {info.language}")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        return conversation_log

# Backward compatibility
VocalMindPipeline = VocalMindPipelineV2

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VocalMind Inference Pipeline v2")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--model", type=str, 
                        default=r"C:\Users\Mohammed Hassan\Zewail\Senior Project\VocalMind-repo\VocalMind\Experiments\Emotion-Recognition\best_model_3class.pt",
                        help="Path to trained emotion model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"CRITICAL: Emotion model not found at {args.model}")
        sys.exit(1)
        
    pipeline = VocalMindPipelineV2(args.model)
    pipeline.process_file(args.audio)
