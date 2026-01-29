
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
import warnings
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, pipeline

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
    # Mappings
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
        # Text encoding
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_out.last_hidden_state[:, 0, :] # CLS token
        
        # Audio encoding
        audio_out = self.audio_encoder(input_values=audio_values)
        audio_embed = torch.mean(audio_out.last_hidden_state, dim=1) # Mean pooling
        
        # Fusion
        fused = torch.cat([self.text_proj(text_embed), self.audio_proj(audio_embed)], dim=1)
        return self.classifier(fused)

# -----------------------------------------------------------------------------
# 2. PIPELINE CLASS (Using Whisper for ASR + Pyannote for Diarization)
# -----------------------------------------------------------------------------
class VocalMindPipelineSimple:
    def __init__(self, emotion_model_path, whisper_model="openai/whisper-base"):
        """
        Pipeline using Whisper for ASR and Pyannote for speaker diarization.
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

        # --- Load Whisper for ASR ---
        print(f"Loading Whisper ASR model ({whisper_model})...")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=0 if torch.cuda.is_available() else -1,
            return_timestamps="word"
        )
        print("Whisper ASR Loaded.")
        
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
        # Move to GPU if available
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        print("Pyannote Diarization Loaded.")
    
    def segment_by_silence(self, audio, sr=16000, min_silence_len=0.5, silence_thresh=-40):
        """
        Simple silence-based segmentation.
        Returns list of (start_sec, end_sec) tuples.
        """
        import librosa
        
        # Convert to dB
        S = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # Find non-silent frames
        non_silent = S_db > silence_thresh
        
        # Get time for each frame
        times = librosa.frames_to_time(np.arange(len(S_db)), sr=sr, hop_length=512)
        
        # Group into segments
        segments = []
        in_segment = False
        start_time = 0
        
        for i, is_voice in enumerate(non_silent):
            if is_voice and not in_segment:
                start_time = times[i]
                in_segment = True
            elif not is_voice and in_segment:
                if times[i] - start_time > 0.3:  # Min segment length
                    segments.append((start_time, times[i]))
                in_segment = False
        
        # Add final segment
        if in_segment:
            segments.append((start_time, times[-1]))
    def detect_silence_gaps(self, audio, sr=16000, min_silence_duration=0.3, silence_thresh_db=-35):
        """
        Detect silence gaps in audio that indicate speaker turns.
        Returns list of (start_time, end_time) for silence segments.
        """
        # Compute RMS energy per frame
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find silent frames
        is_silent = rms_db < silence_thresh_db
        
        # Convert to time
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
        
        # Find silence gaps
        silence_gaps = []
        in_silence = False
        silence_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                silence_start = times[i]
                in_silence = True
            elif not silent and in_silence:
                silence_duration = times[i] - silence_start
                if silence_duration >= min_silence_duration:
                    silence_gaps.append((silence_start, times[i]))
                in_silence = False
        
        return silence_gaps
    
    def process_file(self, audio_path):
        """Runs the full pipeline on a single audio file using Pyannote diarization."""
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return []

        print(f"\nProcessing: {audio_path}")
        
        # Load audio
        print("Loading audio...")
        full_audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(full_audio) / sr
        print(f"Audio duration: {duration:.2f}s")
        
        # Run Pyannote Speaker Diarization
        print("Running Speaker Diarization (Pyannote)...")
        # Prepare audio in format pyannote expects (dict with waveform and sample_rate)
        import torch
        waveform_tensor = torch.from_numpy(full_audio).unsqueeze(0)  # (1, samples)
        audio_input = {"waveform": waveform_tensor, "sample_rate": sr}
        diarization_result = self.diarization_pipeline(audio_input)
        
        # Extract speaker segments from diarization
        # DiarizeOutput has .speaker_diarization which is the Annotation object
        diarization = diarization_result.speaker_diarization
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        
        print(f"Found {len(speaker_segments)} speaker segments")
        
        # Get unique speakers and assign roles (first speaker = Agent typically)
        unique_speakers = list(dict.fromkeys([s['speaker'] for s in speaker_segments]))
        speaker_to_role = {}
        for i, spk in enumerate(unique_speakers):
            speaker_to_role[spk] = "Agent" if i == 0 else "Customer"
        print(f"Speakers detected: {unique_speakers}")
        
        # Run ASR with Whisper
        print("Running ASR...")
        asr_result = self.asr_pipeline(
            audio_path,
            return_timestamps="word",  # Use word-level timestamps for better speaker alignment
            chunk_length_s=30,
            stride_length_s=5
        )
        
        transcript = asr_result.get("text", "")
        asr_chunks = asr_result.get("chunks", [])
        print(f"Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"Transcript: {transcript}")
        
        # Helper function to find speaker at a given time
        def get_speaker_at_time(t):
            for seg in speaker_segments:
                if seg['start'] <= t <= seg['end']:
                    return seg['speaker']
            # Fallback: find closest segment
            min_dist = float('inf')
            closest_speaker = unique_speakers[0] if unique_speakers else None
            for seg in speaker_segments:
                seg_mid = (seg['start'] + seg['end']) / 2
                dist = abs(t - seg_mid)
                if dist < min_dist:
                    min_dist = dist
                    closest_speaker = seg['speaker']
            return closest_speaker
        
        # Group words by speaker (split when speaker changes)
        utterances = []
        current_speaker = None
        current_words = []
        current_start = None
        current_end = None
        
        for chunk in asr_chunks:
            word = chunk.get("text", "").strip()
            ts = chunk.get("timestamp", (0, 0))
            word_start, word_end = ts if ts else (0, 0)
            
            if word_start is None:
                word_start = current_end if current_end else 0
            if word_end is None:
                word_end = word_start + 0.3
            
            if not word:
                continue
            
            # Get speaker for this word based on its midpoint
            word_mid = (word_start + word_end) / 2
            word_speaker = get_speaker_at_time(word_mid)
            
            # If speaker changed, save current utterance and start new one
            if word_speaker != current_speaker and current_words:
                utterances.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_words),
                    'start': current_start,
                    'end': current_end
                })
                current_words = []
                current_start = None
            
            # Add word to current utterance
            if current_start is None:
                current_start = word_start
            current_words.append(word)
            current_end = word_end
            current_speaker = word_speaker
        
        # Don't forget the last utterance
        if current_words:
            utterances.append({
                'speaker': current_speaker,
                'text': ' '.join(current_words),
                'start': current_start,
                'end': current_end
            })
        
        # Post-process: Merge trailing short segments (1-2 words) into the NEXT utterance
        # This fixes issues like "I" at end of Customer being split from Agent's "completely understand"
        final_utterances = []
        i = 0
        while i < len(utterances):
            utt = utterances[i].copy()
            words = utt['text'].split()
            
            # Check if last 1-2 words might belong to next speaker
            if i + 1 < len(utterances) and len(words) > 2:
                next_utt = utterances[i + 1]
                # Common sentence starters that might get mis-attributed
                trailing_starters = ['I', 'We', 'You', 'That', 'This', 'It', 'So', 'But', 'And', 'Or', 'If', 'Yes', 'No', 'Ok', 'Okay']
                
                # Check if last word is a common sentence starter
                last_word = words[-1]
                if last_word in trailing_starters:
                    # Move it to next utterance
                    utt['text'] = ' '.join(words[:-1])
                    utt['end'] = utt['end'] - 0.3  # Approximate
                    next_utt['text'] = last_word + ' ' + next_utt['text']
            
            # Skip empty utterances
            if utt['text'].strip():
                final_utterances.append(utt)
            i += 1
        
        utterances = final_utterances
        
        # Run emotion recognition on each utterance
        print("\nRunning Emotion Recognition...")
        conversation_log = []
        
        for utt in utterances:
            text = utt['text']
            start_sec = utt['start']
            end_sec = utt['end']
            speaker = utt['speaker']
            
            if not text.strip():
                continue
            
            # Slice audio
            start_sample = int(start_sec * 16000)
            end_sample = int(end_sec * 16000)
            audio_segment = full_audio[start_sample:end_sample]
            
            if len(audio_segment) < 1600:  # Skip very short <0.1s
                continue
            
            # Preprocess text
            inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
            
            # Audio feature extraction
            MAX_AUDIO_LEN = 16000 * 6  # 6 seconds max
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
            
            # Map speaker to role using pyannote speaker labels
            role = speaker_to_role.get(speaker, "Unknown")
            
            log_line = f"{role} ({emotion}): {text}"
            print(log_line)
            
            conversation_log.append({
                'speaker': speaker,
                'role': role,
                'emotion': emotion,
                'text': text,
                'timestamp': (start_sec, end_sec)
            })
        
        return conversation_log

# For backward compatibility
VocalMindPipeline = VocalMindPipelineSimple

# -----------------------------------------------------------------------------
# 3. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VocalMind Inference Pipeline (Simplified)")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--model", type=str, 
                        default=r"C:\Users\Mohammed Hassan\Zewail\Senior Project\VocalMind-repo\VocalMind\Experiments\Emotion-Recognition\best_model_3class.pt",
                        help="Path to trained emotion model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"CRITICAL: Emotion model not found at {args.model}")
        sys.exit(1)
        
    pipeline = VocalMindPipelineSimple(args.model)
    pipeline.process_file(args.audio)
