"""
VocalMind Speech Pipeline - Core Transcription Module
======================================================
This module handles:
- Audio loading and preprocessing
- Speaker Diarization (Pyannote 3.1)
- ASR Transcription (faster-whisper)
- Word-Speaker Alignment with overlap handling
- Speaker Role Detection (Agent vs Customer)

Note: Emotion analysis is in speech_pipeline_test.py
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

# -----------------------------------------------------------------------------
# 1. TEXT POST-PROCESSING
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
# 3. OVERLAP DETECTION
# -----------------------------------------------------------------------------
def detect_overlaps(speaker_segments):
    """
    Detect overlapping speech segments where 2+ speakers talk simultaneously.
    Returns list of overlap periods and per-segment overlap metadata.
    """
    overlaps = []
    segment_overlaps = {i: [] for i in range(len(speaker_segments))}
    
    for i, seg1 in enumerate(speaker_segments):
        for j, seg2 in enumerate(speaker_segments[i+1:], start=i+1):
            # Check if segments overlap
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlaps.append({
                    'start': overlap_start,
                    'end': overlap_end,
                    'duration': overlap_duration,
                    'speakers': [seg1['speaker'], seg2['speaker']],
                    'segments': [i, j]
                })
                segment_overlaps[i].append(j)
                segment_overlaps[j].append(i)
    
    # Calculate statistics
    total_overlap_time = sum(o['duration'] for o in overlaps)
    
    return {
        'overlaps': overlaps,
        'segment_overlaps': segment_overlaps,
        'total_overlap_time': total_overlap_time,
        'overlap_count': len(overlaps)
    }

def is_in_overlap(t, overlaps):
    """Check if timestamp t falls within any overlap period."""
    for overlap in overlaps:
        if overlap['start'] <= t <= overlap['end']:
            return True
    return False

# Alignment improvement constants
PAUSE_THRESHOLD_SEC = 0.4  # 400ms gap = natural turn boundary
MIN_SEGMENT_DURATION_SEC = 0.15  # Merge segments shorter than 150ms

def merge_short_segments(speaker_segments, min_duration=MIN_SEGMENT_DURATION_SEC):
    """
    Merge very short speaker segments into neighbors to reduce noise.
    Short segments (< min_duration) are merged into the previous segment.
    """
    if not speaker_segments:
        return speaker_segments
    
    merged = [speaker_segments[0].copy()]
    
    for seg in speaker_segments[1:]:
        seg_duration = seg['end'] - seg['start']
        
        if seg_duration < min_duration:
            # Merge into previous segment (extend its end time)
            merged[-1]['end'] = max(merged[-1]['end'], seg['end'])
        elif seg['speaker'] == merged[-1]['speaker']:
            # Same speaker, just extend
            merged[-1]['end'] = seg['end']
        else:
            # Different speaker, long enough to keep
            merged.append(seg.copy())
    
    return merged

# -----------------------------------------------------------------------------
# 4. SPEAKER ROLE DETECTION
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
# 5. PIPELINE CLASS (faster-whisper + Pyannote)
# -----------------------------------------------------------------------------
class VocalMindPipelineV2:
    def __init__(self, whisper_model="medium"):
        """
        Production pipeline using:
        - faster-whisper medium for ASR (good speed/accuracy balance)
        - Pyannote 3.1 for speaker diarization
        - Overlap detection
        
        Options for whisper_model: "base", "small", "medium", "large-v3"
        """
        # --- Load faster-whisper ---
        self.whisper_model_name = whisper_model  # Store for consistent logging
        print(f"Loading faster-whisper ASR ({whisper_model})...")
        from faster_whisper import WhisperModel
        
        # Use CPU with int8 for compatibility (faster-whisper has cuBLAS 12 incompatibility with CUDA 13)
        # CPU int8 is still very fast for medium model
        self.asr_model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print(f"faster-whisper {whisper_model} Loaded.")
        
        # --- Load Pyannote for Speaker Diarization ---
        print("Loading Pyannote Speaker Diarization model...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Please add your HuggingFace token.")
        
        from pyannote.audio import Pipeline as PyannotePipeline
        
        # Load diarization pipeline
        self.diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
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
        
        diarization = diarization_result
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        
        print(f"Found {len(speaker_segments)} speaker segments")
        
        # Merge very short segments to reduce diarization noise
        original_count = len(speaker_segments)
        speaker_segments = merge_short_segments(speaker_segments)
        if len(speaker_segments) < original_count:
            print(f"Merged short segments: {original_count} -> {len(speaker_segments)}")
        
        unique_speakers = list(dict.fromkeys([s['speaker'] for s in speaker_segments]))
        print(f"Speakers detected: {unique_speakers}")
        
        # --- Overlap Detection ---
        print("\nDetecting overlapping speech...")
        overlap_info = detect_overlaps(speaker_segments)
        overlaps = overlap_info['overlaps']
        segment_overlaps = overlap_info['segment_overlaps']
        total_overlap_time = overlap_info['total_overlap_time']
        overlap_percentage = (total_overlap_time / duration * 100) if duration > 0 else 0
        
        print(f"Found {len(overlaps)} overlapping speech periods")
        print(f"Total overlap time: {total_overlap_time:.2f}s ({overlap_percentage:.1f}% of call)")
        if overlaps:
            print("Sample overlaps:")
            for i, overlap in enumerate(overlaps[:3]):
                print(f"  {i+1}. {overlap['start']:.1f}s-{overlap['end']:.1f}s: {overlap['speakers']} ({overlap['duration']:.2f}s)")
        
        # --- ASR with faster-whisper ---
        print(f"\nRunning ASR (faster-whisper {self.whisper_model_name})...")
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
        
        for i, w in enumerate(all_words):
            word = w['word']
            word_start = w['start']
            word_end = w['end']
            word_prob = w['probability']
            
            if not word:
                continue
            
            word_mid = (word_start + word_end) / 2
            word_speaker = get_speaker_at_time(word_mid)
            
            # --- Speaker Continuity Logic for Overlaps ---
            # If in overlap, prefer continuing current speaker if sentence is incomplete
            if is_in_overlap(word_mid, overlaps) and current_speaker is not None:
                # Check previous word ending
                prev_word = all_words[i-1]['word'] if i > 0 else ""
                prev_end = all_words[i-1]['end'] if i > 0 else 0
                
                # Check for sentence boundary:
                # 1. Punctuation: .?!
                # 2. Pause: gap > threshold
                is_punctuation_end = prev_word.strip()[-1] in ['.', '?', '!', '—', '-'] if prev_word.strip() else True
                is_pause_boundary = (word_start - prev_end) > PAUSE_THRESHOLD_SEC
                
                is_sentence_end = is_punctuation_end or is_pause_boundary
                
                if not is_sentence_end:
                    # Check if current speaker is actually valid for this time (is in the overlap list)
                    valid_speakers = []
                    for ov in overlaps:
                        if ov['start'] <= word_mid <= ov['end']:
                            valid_speakers = ov['speakers']
                            break
                    
                    if current_speaker in valid_speakers:
                        word_speaker = current_speaker
            # ---------------------------------------------
            
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
        trailing_starters = ['I', "I've", "I'm", "I'll", "I'd", 'We', "We're", 'You', "You're", 'That', "That's", 'This', 'It', "It's", 'So', 'But', 'And', 'Or', 'If', 'Yes', 'No', 'Ok', 'Okay']
        
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
        
        # --- Build conversation log ---
        print("\nBuilding conversation log...")
        conversation_log = []
        low_confidence_count = 0
        overlap_affected_count = 0
        
        for utt_idx, utt in enumerate(utterances):
            text = utt['text']
            start_sec = utt['start']
            end_sec = utt['end']
            speaker = utt['speaker']
            confidence = utt.get('confidence', 1.0)
            
            if not text.strip():
                continue
            
            # Check if this utterance overlaps with any other speakers
            has_overlap = False
            overlap_with = []
            overlap_duration = 0
            
            for overlap in overlaps:
                # Check if utterance overlaps with detected overlap period
                utt_overlap_start = max(start_sec, overlap['start'])
                utt_overlap_end = min(end_sec, overlap['end'])
                
                if utt_overlap_start < utt_overlap_end:
                    has_overlap = True
                    overlap_duration += (utt_overlap_end - utt_overlap_start)
                    # Get the other speaker(s) in this overlap
                    other_speakers = [s for s in overlap['speakers'] if s != speaker]
                    overlap_with.extend(other_speakers)
            
            overlap_with = list(dict.fromkeys(overlap_with))  # Remove duplicates
            if has_overlap:
                overlap_affected_count += 1
            
            # Flag low confidence
            confidence_flag = ""
            if confidence < 0.7:
                confidence_flag = " [LOW CONF]"
                low_confidence_count += 1
            
            # Flag overlaps
            overlap_flag = ""
            if has_overlap:
                overlap_flag = f" [OVERLAP with {', '.join(overlap_with)}]"
            
            role = speaker_to_role.get(speaker, "Unknown")
            
            log_line = f"{role}: {text}{confidence_flag}{overlap_flag}"
            print(log_line)
            
            conversation_log.append({
                'speaker': speaker,
                'role': role,
                'text': text,
                'confidence': confidence,
                'timestamp': (start_sec, end_sec),
                'has_overlap': has_overlap,
                'overlap_with': overlap_with if has_overlap else None,
                'overlap_duration': overlap_duration if has_overlap else 0
            })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total utterances: {len(conversation_log)}")
        print(f"Low confidence segments: {low_confidence_count}")
        print(f"Overlap-affected utterances: {overlap_affected_count} ({overlap_affected_count/len(conversation_log)*100:.1f}%)" if conversation_log else "")
        print(f"Total overlap periods: {len(overlaps)}")
        print(f"Total overlap time: {total_overlap_time:.2f}s ({overlap_percentage:.1f}% of call)")
        print(f"Language: {info.language}")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        return conversation_log

# Backward compatibility
VocalMindPipeline = VocalMindPipelineV2

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VocalMind Inference Pipeline v2")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--whisper-model", type=str, default="medium",
                        choices=["base", "small", "medium", "large-v3"],
                        help="Whisper model size")
    
    args = parser.parse_args()
        
    pipeline = VocalMindPipelineV2(args.whisper_model)
    pipeline.process_file(args.audio)
