"""
VocalMind Speech Pipeline V2 (Combined)
=======================================
Combines:
1. WhisperX (GPU-accelerated ASR + Alignment + Diarization)
2. Advanced Multimodal Emotion Fusion (Text + Audio)
3. Speaker Role Classification (Agent vs Customer) 
4. Memory Optimization (Sequential Model Loading)

Usage:
    python main_v2.py --audio path/to/audio.mp3
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
import re
import numpy as np
import torch
import librosa
import transformers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==============================================================================
# 0. COMPATIBILITY PATCHES
# ==============================================================================
print("Applying compatibility patches...")

# PATCH 1: torch.load weights_only fix
try:
    original_load = torch.load
    def custom_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = custom_load
    print("[OK] torch.load patch applied")
except Exception:
    pass

# PATCH 2: torchaudio 2.11+ missing attributes (AudioMetaData, backends)
try:
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding
        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    print("[OK] torchaudio compatibility patch applied")
except ImportError:
    pass

# PATCH 3: huggingface_hub use_auth_token deprecation (for Pyannote)
try:
    import huggingface_hub
    original_hf_hub_download = huggingface_hub.hf_hub_download
    def patched_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return original_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = patched_hf_hub_download
    print("[OK] huggingface_hub patch applied")
except ImportError:
    pass

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# ==================== EASY AUDIO FILE CONFIGURATION ====================
# Change this path to test different audio files without command-line args
DEFAULT_AUDIO_FILE = "../Voice-Generation/generated_audio/medium_overlap.mp3"
# ========================================================================

# Environment loading
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("⚠ WARNING: HF_TOKEN not found in .env")

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Device: {DEVICE} ({COMPUTE_TYPE})")

# Suppress warnings
warnings.filterwarnings("ignore")
if 'transformers' in sys.modules:
    import transformers
    transformers.logging.set_verbosity_error()

# ==============================================================================
# 2. EMOTION MODELS & LOGIC (From run_demo.py)
# ==============================================================================

# Emotion Constants
TEXT_EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # VAD-based audio emotion

# GoEmotions 28-class to 3-class sentiment mapping
EMOTION_TO_SENTIMENT = {
    # Negative emotions
    "anger": "Negative", "annoyance": "Negative", "disappointment": "Negative",
    "disapproval": "Negative", "disgust": "Negative", "embarrassment": "Negative",
    "fear": "Negative", "grief": "Negative", "nervousness": "Negative",
    "remorse": "Negative", "sadness": "Negative",
    
    # Positive emotions
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive",
    "caring": "Positive", "desire": "Positive", "excitement": "Positive",
    "gratitude": "Positive", "joy": "Positive", "love": "Positive",
    "optimism": "Positive", "pride": "Positive", "relief": "Positive",
    
    # Neutral emotions
    "confusion": "Neutral", "curiosity": "Neutral", "neutral": "Neutral",
    "realization": "Neutral", "surprise": "Neutral"
}

# Audio emotion mapping (emotion2vec uses different labels)
AUDIO_TO_SENTIMENT = {
    "angry": "Negative", "disgusted": "Negative", "fearful": "Negative", "sad": "Negative",
    "happy": "Positive", "surprised": "Neutral", "neutral": "Neutral", "other": "Neutral"
}

POSITIVE_PHRASE_PATTERNS = [
    "thank you", "thanks", "appreciate", "grateful", "have a nice day", "great day",
    "take care", "good bye", "welcome", "my pleasure", "happy to help", "sounds good",
    "perfect", "great", "wonderful", "excellent"
]

NEGATIVE_PHRASE_PATTERNS = [
    "upset", "frustrated", "angry", "annoyed", "disappointed", "unhappy",
    "terrible", "worst", "horrible", "ridiculous", "unacceptable"
]

EMPATHY_PHRASE_PATTERNS = [
    "understand your", "sorry to hear", "apologize for", "must be frustrating",
    "see why", "help you with", "figure this out", "resolve this"
]

# ==============================================================================
# FUSION CONFIGURATION (Tune these for your domain)
# ==============================================================================
FUSION_CONFIG = {
    # Confidence thresholds
    'text_confident_threshold': 0.75,
    'audio_confident_threshold': 0.75,
    'audio_override_threshold': 0.80,  # For AGENT: only trust audio if >= 80%
    
    # Base weights for weighted voting
    'text_weight': 0.6,
    'audio_weight': 0.4,
    
    # Short segment handling (<threshold seconds)
    'short_segment_threshold': 0.7,  # Reduced from 0.8 for stricter handling
    'short_segment_text_weight': 0.80,
    'short_segment_audio_weight': 0.20,
    
    # Agent role audio penalty (agents sound "concerned" which is misread as negative)
    'agent_audio_weight_multiplier': 0.5,  # Reduce audio weight by 50% for agents
    
    # SURPRISE emotion mapping (context-aware)
    'surprise_default_sentiment': 'Neutral',
}

# Explicit emotion-to-polarity mapping (simplified and fixed)
EMOTION_POLARITY = {
    # Positive emotions
    'joy': 'Positive', 'caring': 'Positive', 'gratitude': 'Positive', 
    'approval': 'Positive', 'happy': 'Positive', 'admiration': 'Positive',
    'amusement': 'Positive', 'excitement': 'Positive', 'love': 'Positive',
    'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive',
    'desire': 'Positive',
    
    # Neutral emotions
    'neutral': 'Neutral', 'curiosity': 'Neutral', 'surprise': 'Neutral',
    'surprised': 'Neutral', 'realization': 'Neutral', 'confusion': 'Neutral',
    'other': 'Neutral',
    
    # Negative emotions
    'anger': 'Negative', 'angry': 'Negative', 'annoyance': 'Negative',
    'disgust': 'Negative', 'disgusted': 'Negative', 'fear': 'Negative',
    'fearful': 'Negative', 'sadness': 'Negative', 'sad': 'Negative',
    'disapproval': 'Negative', 'disappointment': 'Negative',
    'embarrassment': 'Negative', 'grief': 'Negative', 'nervousness': 'Negative',
    'remorse': 'Negative'
}

def get_polarity(emotion: str) -> str:
    """Get polarity from emotion using explicit mapping."""
    return EMOTION_POLARITY.get(emotion.lower(), 'Neutral')



class RegressionHead(torch.nn.Module):
# ... (existing classes) ...


    """Head for audeering audio emotion model"""
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(transformers.Wav2Vec2PreTrainedModel):
    """Audio emotion model wrapper"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = transformers.Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

class AudioEmotionClassifier:
    """Wav2Vec2-based audio emotion classifier with VAD dimensional output."""
    
    def __init__(self):
        print(f"Loading Audio Emotion Model: {AUDIO_EMOTION_MODEL}...")
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(AUDIO_EMOTION_MODEL)
        self.model = EmotionModel.from_pretrained(AUDIO_EMOTION_MODEL)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        print("[OK] Audio Emotion Model Loaded")

    def _dimensions_to_emotion(self, arousal, valence, dominance):
        """Map VAD dimensions to emotion label."""
        v_low, v_high, a_low, a_high = 0.4, 0.6, 0.4, 0.6
        if arousal > 0.7:
            if valence > 0.5: return "happy", valence * 0.9 + arousal * 0.1
            else: return ("angry" if dominance > 0.5 else "fearful"), (1 - valence) * 0.8 + arousal * 0.2
        if valence > v_high: return ("happy" if arousal > 0.5 else "neutral"), valence
        if valence < v_low:
            if arousal > a_high: return ("angry" if dominance > 0.5 else "fearful"), 1 - valence
            elif arousal < a_low: return "sad", 1 - valence
            else: return "disgusted", 1 - valence
        return "neutral", 0.5 + abs(valence - 0.5)

    def _calculate_emotion_scores(self, arousal: float, valence: float, dominance: float) -> Dict:
        """Calculate approximate scores for all emotions based on dimensional values."""
        def gaussian_score(v, a, v_center, a_center, sigma=0.3):
            dist = ((v - v_center)**2 + (a - a_center)**2) ** 0.5
            return max(0, 1 - dist / sigma)
        
        scores = {
            'angry': gaussian_score(valence, arousal, 0.2, 0.8) * (0.5 + dominance * 0.5),
            'disgusted': gaussian_score(valence, arousal, 0.3, 0.5),
            'fearful': gaussian_score(valence, arousal, 0.3, 0.8) * (1.5 - dominance),
            'happy': gaussian_score(valence, arousal, 0.8, 0.7),
            'neutral': gaussian_score(valence, arousal, 0.5, 0.5),
            'sad': gaussian_score(valence, arousal, 0.2, 0.2),
            'surprised': gaussian_score(valence, arousal, 0.6, 0.9)
        }
        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}

    def predict(self, audio_array, sr=16000):
        """Predict emotion from audio segment."""
        if len(audio_array) < sr * 1.0:
            audio_array = np.pad(audio_array, (0, int(sr * 1.0) - len(audio_array)), mode='constant')
        
        # Process
        inputs = self.processor(audio_array, sampling_rate=sr)
        input_values = torch.tensor(inputs['input_values'][0]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, logits = self.model(input_values)
            dims = logits[0].cpu().numpy()
            
        arousal, dominance, valence = float(dims[0]), float(dims[1]), float(dims[2])
        emotion, conf = self._dimensions_to_emotion(arousal, valence, dominance)
        all_scores = self._calculate_emotion_scores(arousal, valence, dominance)
        
        return {
            'emotion': emotion,
            'confidence': conf,
            'sentiment': AUDIO_TO_SENTIMENT.get(emotion, 'Neutral'),
            'dimensions': {'arousal': arousal, 'dominance': dominance, 'valence': valence},
            'all_scores': all_scores
        }

class TextEmotionClassifier:
    def __init__(self):
        print(f"Loading Text Emotion Model: {TEXT_EMOTION_MODEL}...")
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model=TEXT_EMOTION_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None
        )
        print("[OK] Text Emotion Model Loaded")

    def predict(self, text):
        if not text.strip(): 
            return {'emotion': 'neutral', 'confidence': 0.0, 'sentiment': 'Neutral', 'all_scores': {}, 'corrected': False}
        
        # Normalize
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)', r'$\1', text)
        text = " ".join(text.split())

        result = self.classifier(text)
        top = max(result[0], key=lambda x: x['score'])
        all_scores = {r['label']: r['score'] for r in result[0]}
        
        return {
            'emotion': top['label'],
            'confidence': top['score'],
            'sentiment': EMOTION_TO_SENTIMENT.get(top['label'], 'Neutral'),
            'all_scores': all_scores,
            'corrected': False,
            'original_emotion': None
        }

def apply_fusion_logic(text_res, audio_res, text_content, segment_duration=None, speaker_role=None):
    """Combine text and audio predictions with business logic corrections.
    
    Refined fusion rules for customer support:
    1. AGENT segments: reduce audio influence by 50%, prefer text unless audio >= 80%
    2. Short utterances (<0.7s): text_weight=0.8, audio_weight=0.2
    3. Explicit polarity mapping for all emotions
    4. SURPRISE: default Neutral, only Positive if audio=happy or text in [joy, gratitude]
    5. Disgust suppression for AGENT when text is neutral/curiosity
    
    Log sources: TEXT >=75%, AUDIO >=75%, TEXT weighted, BOTH AGREE
    """
    cfg = FUSION_CONFIG
    
    # Extract values
    text_lower = text_content.lower()
    t_emotion, t_conf = text_res['emotion'], text_res['confidence']
    a_emotion, a_conf = audio_res['emotion'], audio_res['confidence']
    segment_duration = segment_duration if segment_duration else 1.0
    speaker_role = speaker_role if speaker_role else 'UNKNOWN'
    
    original_emotion = t_emotion
    was_corrected = False
    
    # =========================================================================
    # STEP 1: Rule-based text corrections (empathy phrases, etc.)
    # =========================================================================
    
    # EMPATHY CHECK: Agent saying "understand your frustration" is NOT anger
    if t_emotion in ['anger', 'sadness', 'disgust', 'fear', 'annoyance', 'disapproval']:
        for phrase in EMPATHY_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion = 'neutral'
                t_conf = max(t_conf, 0.85)
                was_corrected = True
                break

    # Correct positive phrases misclassified as negative
    if t_emotion in ['fear', 'anger', 'disgust', 'sadness', 'annoyance']:
        for phrase in POSITIVE_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion = 'neutral'
                t_conf = max(t_conf, 0.7)
                was_corrected = True
                break
    
    # =========================================================================
    # STEP 2: Get polarities using explicit mapping
    # =========================================================================
    t_polarity = get_polarity(t_emotion)
    a_polarity = get_polarity(a_emotion)
    
    # =========================================================================
    # STEP 3: SURPRISE handling (special rule)
    # Default: Neutral. Only Positive if audio=happy or text in [joy, gratitude]
    # =========================================================================
    if t_emotion == 'surprise' or t_emotion == 'surprised':
        if a_emotion == 'happy' or t_emotion in ['joy', 'gratitude']:
            t_polarity = 'Positive'
        else:
            t_polarity = 'Neutral'
    
    # =========================================================================
    # STEP 4: Calculate effective weights based on role and duration
    # =========================================================================
    text_weight = cfg['text_weight']
    audio_weight = cfg['audio_weight']
    is_short_segment = segment_duration < cfg['short_segment_threshold']
    is_agent = speaker_role == 'AGENT'
    
    # Short utterances (<0.7s): text_weight=0.8, audio_weight=0.2
    if is_short_segment:
        text_weight = cfg['short_segment_text_weight']
        audio_weight = cfg['short_segment_audio_weight']
    
    # AGENT segments: reduce audio weight by 50%
    if is_agent:
        audio_weight = audio_weight * cfg['agent_audio_weight_multiplier']
    
    # Effective audio confidence (for threshold checks)
    a_conf_effective = a_conf
    if is_short_segment:
        a_conf_effective *= 0.5  # Penalize short segment audio
    
    # =========================================================================
    # STEP 5: Disgust false-positive suppression for AGENT
    # If AGENT + audio=disgusted + text is neutral/curiosity -> override to neutral
    # =========================================================================
    disgust_suppressed = False
    if is_agent and a_emotion == 'disgusted' and t_polarity == 'Neutral':
        disgust_suppressed = True
    
    # =========================================================================
    # STEP 6: Deterministic Fusion Logic
    # =========================================================================
    
    # Rule 1: Text is confident (>=75%) -> trust text
    if t_conf >= cfg['text_confident_threshold']:
        fused_emotion = t_emotion
        fused_polarity = t_polarity
        fused_conf = t_conf
        source = "TEXT >=75%"
    
    # Rule 2: Audio is confident (>=75% effective, or >=80% for AGENT override)
    elif (not is_agent and a_conf_effective >= cfg['audio_confident_threshold']) or \
         (is_agent and a_conf >= cfg['audio_override_threshold']):
        # Apply disgust suppression for AGENT
        if disgust_suppressed:
            fused_emotion = 'neutral'
            fused_polarity = 'Neutral'
            fused_conf = t_conf
            source = "TEXT >=75%"  # Override to text due to disgust suppression
        else:
            fused_emotion = a_emotion
            fused_polarity = a_polarity
            fused_conf = a_conf
            source = "AUDIO >=75%"
    
    # Rule 3: Both agree on polarity -> boost confidence, use text label
    elif t_polarity == a_polarity:
        fused_emotion = t_emotion
        fused_polarity = t_polarity
        fused_conf = min(1.0, (t_conf + a_conf) / 1.5)
        source = "BOTH AGREE"
    
    # Rule 4: Disagreement -> weighted vote
    else:
        weighted_text = t_conf * text_weight
        weighted_audio = a_conf_effective * audio_weight
        
        if weighted_text >= weighted_audio:
            fused_emotion = t_emotion
            fused_polarity = t_polarity
            fused_conf = t_conf
            source = "TEXT weighted"
        else:
            # Apply disgust suppression for AGENT even in weighted case
            if disgust_suppressed:
                fused_emotion = 'neutral'
                fused_polarity = 'Neutral'
                fused_conf = t_conf
                source = "TEXT weighted"  # Suppressed disgust
            else:
                fused_emotion = a_emotion
                fused_polarity = a_polarity
                fused_conf = a_conf
                source = "AUDIO weighted"
        
        # Fallback: if both very low (<50%), default to neutral
        if t_conf < 0.50 and a_conf_effective < 0.50:
            fused_emotion = 'neutral'
            fused_polarity = 'Neutral'
            fused_conf = max(t_conf, a_conf_effective)
            source = "TEXT weighted"  # Low confidence fallback

    return {
        'emotion': fused_emotion,
        'sentiment': fused_polarity,
        'confidence': fused_conf,
        'source': source,
        'corrected': was_corrected,
        'original_emotion': original_emotion if was_corrected else None,
        'text_details': {
            'emotion': t_emotion, 
            'confidence': t_conf,
            'all_scores': text_res.get('all_scores', {})
        },
        'audio_details': {
            'emotion': a_emotion,
            'confidence': a_conf,
            'dimensions': audio_res.get('dimensions', {}),
            'all_scores': audio_res.get('all_scores', {})
        }
    }

# ==============================================================================
# 2b. SPEAKER ROLE CLASSIFIER (From main.py)
# ==============================================================================
class SpeakerRoleClassifier:
    """
    State-of-the-art speaker role identification using:
    1. LLM-based zero-shot classification (BART)
    2. Linguistic pattern analysis
    3. Sentiment profiling
    4. Question type detection
    5. Conversational dynamics
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        print("   Initializing speaker role classifier...")
        
        from transformers import pipeline
        
        # Zero-shot classifier
        print("   [1/3] Loading zero-shot LLM...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            zero_shot_model_name = "facebook/bart-large-mnli"
            zero_shot_tokenizer = AutoTokenizer.from_pretrained(zero_shot_model_name)
            zero_shot_model = AutoModelForSequenceClassification.from_pretrained(
                zero_shot_model_name,
                use_safetensors=True
            )
            
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=zero_shot_model,
                tokenizer=zero_shot_tokenizer,
                device=0 if device == "cuda" else -1
            )
        except Exception as e:
            print(f"      ⚠ Zero-shot model unavailable, using heuristics")
            self.zero_shot_classifier = None
        
        # Sentiment analyzer
        print("   [2/3] Loading sentiment analyzer...")
        try:
            # Try with safe loading (safetensors)
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model_name,
                use_safetensors=True  # Force safe format
            )
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                tokenizer=sentiment_tokenizer,
                device=0 if device == "cuda" else -1
            )
        except Exception as e:
            print(f"      ⚠ Sentiment analyzer unavailable, using fallback")
            self.sentiment_analyzer = None
        
        # Question classifier
        print("   [3/3] Loading question classifier...")
        try:
            self.question_classifier = pipeline(
                "text-classification",
                model="shahrukhx01/question-vs-statement-classifier",
                device=0 if device == "cuda" else -1
            )
        except:
            print("      ⚠ Question classifier unavailable, using fallback")
            self.question_classifier = None
        
        # Linguistic patterns
        self.agent_indicators = {
            'formal_language': [
                'understand', 'assist', 'help', 'support', 'service',
                'appreciate', 'thank you for', 'apologize', 'certainly',
                'i can', 'let me', 'i will', 'happy to', 'glad to'
            ],
            'procedural': [
                'need to', 'going to', 'will need', 'can i get',
                'let me check', 'looking at', 'according to', 'shows that'
            ],
            'empathy': [
                'understand your', 'i see', 'that must be', 'frustrating',
                'concerning', 'definitely'
            ]
        }
        
        self.customer_indicators = {
            'problem_framing': [
                'my', 'i have', 'i need', 'issue', 'problem', 'wrong',
                'not working', 'help me', 'can you', 'why is'
            ],
            'frustration': [
                'upset', 'angry', 'frustrated', 'ridiculous', 'terrible',
                'disappointed', 'unacceptable'
            ]
        }
        
        print("   [OK] Speaker role classifier ready\n")
    
    def extract_linguistic_features(self, text: str, speaker_id: str, 
                                   all_segments: List[Dict]) -> Dict[str, float]:
        """Extract linguistic and conversational features"""
        
        text_lower = text.lower()
        features = {}
        
        # Helper for saturation scoring
        def score_indicators(indicators, text):
            count = sum(1 for phrase in indicators if phrase in text)
            # Presence of 2+ indicators = 100% confidence, 1 = 60%
            return min(1.0, count * 0.6)
        
        # Formal language
        features['formal_language_score'] = score_indicators(
            self.agent_indicators['formal_language'], text_lower
        )
        
        # Procedural language
        features['procedural_score'] = score_indicators(
            self.agent_indicators['procedural'], text_lower
        )
        
        # Empathy markers
        features['empathy_score'] = score_indicators(
            self.agent_indicators['empathy'], text_lower
        )
        
        # Problem framing (customer)
        features['problem_framing_score'] = score_indicators(
            self.customer_indicators['problem_framing'], text_lower
        )
        
        # Frustration markers (customer)
        features['frustration_score'] = score_indicators(
            self.customer_indicators['frustration'], text_lower
        )
        
        # Question density (Keep as density but scale it)
        question_marks = text.count('?')
        features['question_density'] = min(1.0, (question_marks / max(len(text.split()), 1)) * 10)
        
        # Average utterance length (Normalize: 20 words = 1.0)
        speaker_segments = [s for s in all_segments if s.get('speaker') == speaker_id]
        if speaker_segments:
            avg_len = np.mean([len(s.get('text', '').split()) for s in speaker_segments])
            features['avg_utterance_length'] = min(1.0, avg_len / 20.0)
        else:
            features['avg_utterance_length'] = 0.0
        
        # First speaker (Stronger signal for Agent in call center)
        speaker_turns = [s.get('speaker') for s in all_segments]
        first_speaker = speaker_turns[0] if speaker_turns else None
        features['is_first_speaker'] = 1.0 if speaker_id == first_speaker else 0.0
        
        return features
    
    def classify_with_zero_shot(self, text: str) -> Dict[str, float]:
        """LLM-based zero-shot classification"""
        
        if self.zero_shot_classifier is None:
            # Fallback: rule-based heuristics
            text_lower = text.lower()
            agent_score = 0.5
            
            # Agent indicators
            if any(phrase in text_lower for phrase in ['thank you for calling', 'how can i help', 
                                                        'let me check', 'i understand', 'my name is']):
                agent_score += 0.3
            if any(phrase in text_lower for phrase in ['looking at your', 'according to', 
                                                        'i can see', 'shows that']):
                agent_score += 0.2
            
            # Customer indicators
            if any(phrase in text_lower for phrase in ['my bill', 'i have a problem', 
                                                        'i need help', "i'm upset"]):
                agent_score -= 0.3
            
            agent_score = max(0.1, min(0.9, agent_score))
            return {'agent_prob': agent_score, 'customer_prob': 1 - agent_score}
        
        candidate_labels = ["customer service representative", "customer with a problem"]
        
        try:
            result = self.zero_shot_classifier(
                text[:1000], candidate_labels, multi_label=False
            )
            agent_prob = result['scores'][0] if result['labels'][0] == candidate_labels[0] else result['scores'][1]
            return {'agent_prob': agent_prob, 'customer_prob': 1 - agent_prob}
        except:
            return {'agent_prob': 0.5, 'customer_prob': 0.5}
    
    def analyze_sentiment_pattern(self, text: str) -> Dict[str, float]:
        """Sentiment analysis helper"""
        if self.sentiment_analyzer is None:
            return {'sentiment_agent_score': 0.5, 'sentiment_confidence': 0.0}
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            label = result['label'].lower()
            score = result['score']
            
            if label == 'neutral': agent_indicator = 0.7
            elif label == 'positive': agent_indicator = 0.6
            else: agent_indicator = 0.2
            
            return {'sentiment_agent_score': agent_indicator, 'sentiment_confidence': score}
        except:
            return {'sentiment_agent_score': 0.5, 'sentiment_confidence': 0.0}
    
    def detect_question_pattern(self, text: str) -> Dict[str, float]:
        """Question type detection"""
        if self.question_classifier is None:
            has_question = '?' in text
            return {'question_agent_score': 0.6 if has_question else 0.5, 'question_confidence': 0.5}
        
        try:
            result = self.question_classifier(text[:512])[0]
            is_question = result['label'] == 'LABEL_1'
            confidence = result['score']
            
            if is_question:
                text_lower = text.lower()
                clarifying = any(p in text_lower for p in ['can i get', 'may i have', 'could you confirm'])
                help_seeking = any(p in text_lower for p in ['why is', 'how do i', 'can you help'])
                
                if clarifying: return {'question_agent_score': 0.8, 'question_confidence': confidence}
                elif help_seeking: return {'question_agent_score': 0.2, 'question_confidence': confidence}
            
            return {'question_agent_score': 0.5, 'question_confidence': confidence}
        except:
            return {'question_agent_score': 0.5, 'question_confidence': 0.0}
    
    def ensemble_classification(self, features: Dict[str, float], 
                               zero_shot: Dict[str, float],
                               sentiment: Dict[str, float],
                               question: Dict[str, float]) -> Tuple[str, float]:
        """Ensemble fusion"""
        ling_agent_prob = (
            features['formal_language_score'] * 0.25 +
            features['procedural_score'] * 0.25 +
            features['empathy_score'] * 0.15 +
            (1 - features['problem_framing_score']) * 0.15 +
            (1 - features['frustration_score']) * 0.10 +
            features['is_first_speaker'] * 0.10
        )
        agent_probability = (
            zero_shot['agent_prob'] * 0.35 +
            ling_agent_prob * 0.30 +
            sentiment['sentiment_agent_score'] * 0.20 +
            question['question_agent_score'] * 0.15
        )
        
        if agent_probability > 0.55: return "AGENT", agent_probability
        elif agent_probability < 0.45: return "CUSTOMER", 1 - agent_probability
        else:
            if features['is_first_speaker'] > 0.5 and features['formal_language_score'] > 0.05:
                return "AGENT", 0.7
            else:
                return "CUSTOMER", 0.7
    
    def classify_speaker_role(self, speaker_id: str, segments: List[Dict]) -> Tuple[str, float, Dict]:
        speaker_segments = [s for s in segments if s.get('speaker') == speaker_id]
        if not speaker_segments: return "UNKNOWN", 0.0, {}
        
        full_text = " ".join([s.get('text', '') for s in speaker_segments])
        linguistic = self.extract_linguistic_features(full_text, speaker_id, segments)
        zero_shot = self.classify_with_zero_shot(full_text)
        sentiment = self.analyze_sentiment_pattern(full_text)
        question = self.detect_question_pattern(full_text)
        
        role, conf = self.ensemble_classification(linguistic, zero_shot, sentiment, question)
        
        return role, conf, {
            'ling': linguistic, 'zero': zero_shot, 'sent': sentiment, 'quest': question
        }

    def classify_all_speakers(self, segments: List[Dict]) -> Dict[str, str]:
        speakers = list(set(s.get('speaker') for s in segments if s.get('speaker')))
        results = {}
        print("   Classifying speaker roles...")
        for speaker in speakers:
            role, conf, details = self.classify_speaker_role(speaker, segments)
            results[speaker] = role
            print(f"   {speaker}: {role} (confidence: {conf:.2f})")
                  
        return results

def detect_overlaps(segments, threshold=0.1):
    """
    Detect overlapping segments in a sorted list of segments.
    Mark segments with 'overlap': True if they intersect with others.
    """
    # Sort just in case
    segments = sorted(segments, key=lambda x: x['start'])
    
    for i in range(len(segments)):
        if 'overlap' not in segments[i]:
            segments[i]['overlap'] = False
            
    for i in range(len(segments)):
        curr = segments[i]
        
        # Check subsequent segments
        for j in range(i + 1, len(segments)):
            next_seg = segments[j]
            
            # If next segment starts after current ends, no more overlap possible (sorted)
            if next_seg['start'] >= curr['end']:
                break
                
            # Overlap found
            curr['overlap'] = True
            next_seg['overlap'] = True
            
    return segments

# ==============================================================================
# 3. MAIN PIPELINE (Memory Optimized)
# ==============================================================================
class CombinedPipeline:
    def __init__(self, whisper_model_size="large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing Combined Pipeline (Sequential Loading) ===")
        # Models will be loaded on demand to save VRAM
        
    def _load_whisper_and_diarize(self):
        print(">> Loading ASR & Diarization Models...")
        import whisperx
        from whisperx.diarize import DiarizationPipeline
        
        # 1. ASR
        asr_model = whisperx.load_model(self.whisper_model_size, DEVICE, compute_type=COMPUTE_TYPE)
        print("     [OK] WhisperX Loaded")
        
        # 2. Diarization
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("     [OK] Diarization Loaded")
        
        return whisperx, asr_model, diarize_model

    def _load_analysis_models(self):
        print(">> Loading Analysis Models (Emotion & Role)...")
        return {
            'text': TextEmotionClassifier(),
            'audio': AudioEmotionClassifier(),
            'role': SpeakerRoleClassifier(device=DEVICE)
        }

    def process(self, audio_path, language=None):
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return
            
        print(f"Processing: {audio_path}")
        
        # PHASE 1: ASR & DIARIZATION
        whisperx, asr_model, diarize_model = self._load_whisper_and_diarize()
        
        # 1. Transcribe
        print(">> Step 1: Transcribing...")
        # Load audio using WhisperX
        audio = whisperx.load_audio(audio_path)
        batch_size = 16
        result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
        language = result["language"]
        print(f"   Language: {language}")
        
        # 2. Align
        print(">> Step 2: Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        # 3. Diarize
        print(">> Step 3: Diarizing...")
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 4. Overlap Detection
        print(">> Step 3b: Detecting Overlaps...")
        result["segments"] = detect_overlaps(result["segments"])
        
        # CLEANUP PHASE 1
        print(">> Cleaning up ASR models to free VRAM...")
        del asr_model
        del diarize_model
        del model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Memory Freed")
            
        # PHASE 2: ANALYSIS
        print(">> Step 4a: Identifying Speaker Roles...")
        # Load analysis models now that VRAM is free
        analysis_models = self._load_analysis_models()
        role_classifier = analysis_models['role']
        text_classifier = analysis_models['text']
        audio_classifier = analysis_models['audio']

        # Flatten segments for classification context
        all_segments_flat = []
        for segment in result["segments"]:
            all_segments_flat.append({
                "text": segment["text"].strip(),
                "speaker": segment.get("speaker", "UNKNOWN")
            })
        speaker_roles = role_classifier.classify_all_speakers(all_segments_flat)
        
        # 5. Emotion Analysis
        print(">> Step 5: Analyzing Emotions (Multimodal Fusion)...")
        final_segments = []
        
        # Load audio for feature extraction
        full_audio_librosa, sr = librosa.load(audio_path, sr=16000)
        
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            speaker = segment.get("speaker", "UNKNOWN")
            role = speaker_roles.get(speaker, "UNKNOWN")
            
            # Extract Audio Config
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            audio_segment = full_audio_librosa[start_sample:end_sample]
            
            # Predict
            text_res = text_classifier.predict(text)
            audio_res = audio_classifier.predict(audio_segment)
            
            # Fuse (pass segment duration and speaker role for adjustments)
            segment_duration = end - start
            fusion = apply_fusion_logic(text_res, audio_res, text, segment_duration, speaker_role=role)
            
            final_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "role": role,
                "text": text,
                "emotion_analysis": fusion
            })
            
            self._print_utterance(
                speaker, role, text, start, end,
                text_res, audio_res, fusion,
                is_overlap=segment.get('overlap', False)
            )
            
        return final_segments

    def _print_utterance(self, speaker, role, text, start, end, text_res, audio_res, fusion, is_overlap=False):
        """Print detailed utterance with both text and audio emotions."""
        overlap_tag = " [OVERLAP]" if is_overlap else ""
        
        # Text emotion details
        t_emo = text_res['emotion']
        t_conf = text_res['confidence']
        t_sent = text_res['sentiment']
        
        # Audio emotion details
        a_emo = audio_res['emotion']
        a_conf = audio_res['confidence']
        a_sent = audio_res['sentiment']
        dims = audio_res.get('dimensions', {})
        dims_str = ""
        if dims:
            dims_str = f" [V:{dims.get('valence', 0):.2f} A:{dims.get('arousal', 0):.2f} D:{dims.get('dominance', 0):.2f}]"
        
        # Fused result
        f_emo = fusion['emotion']
        f_conf = fusion['confidence']
        f_sent = fusion['sentiment']
        f_src = fusion['source']
        
        # Correction info
        corrected_str = ""
        if fusion.get('corrected') and fusion.get('original_emotion'):
            corrected_str = f" (was: {fusion['original_emotion']} -> corrected)"
        
        # Print
        print(f"\n[{start:.1f}s-{end:.1f}s] {role}{overlap_tag}")
        print(f"   Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"   Text Emotion:  {t_emo:12s} ({t_conf:5.1%}) -> {t_sent}{corrected_str}")
        print(f"   Audio Emotion: {a_emo:12s} ({a_conf:5.1%}) -> {a_sent}{dims_str}")
        
        # Show fusion decision with new deterministic labels
        source_labels = {
            'text-confident': 'TEXT >=75%',
            'audio-confident': 'AUDIO >=75%',
            'both-agree': 'BOTH AGREE',
            'text-weighted': 'TEXT weighted',
            'audio-weighted': 'AUDIO weighted',
            'low-conf-neutral': 'LOW CONF->NEUTRAL'
        }
        label = source_labels.get(f_src, f_src.upper())
        print(f"   >> Fused:      {f_emo:12s} ({f_conf:5.1%}) -> {f_sent} [{label}]")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON with full emotion scores for analysis."""
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert_numpy(obj)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deep_convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalMind Combined Pipeline")
    parser.add_argument("--audio", type=str, default=DEFAULT_AUDIO_FILE, help=f"Input audio file (default: {DEFAULT_AUDIO_FILE})")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g. en, ar) to skip detection")
    parser.add_argument("--output", type=str, help="Save results to JSON file (includes raw emotion scores)")
    args = parser.parse_args()
    
    # Support running without args (uses DEFAULT_AUDIO_FILE)
    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)
    
    pipeline = CombinedPipeline()
    results = pipeline.process(args.audio, language=args.language)
    
    if args.output and results:
        save_results_json(results, args.output)
    
    print("\n" + "="*50)
    print("DONE.")
