"""
VocalMind Pipeline v3 — Emotion Model Comparison
=================================================

Purpose:
    Compare new emotion models against the current v2 models to determine
    which combination produces more reliable results for customer-service
    audio analysis.

Current models (v2):
    Text:  SamLowe/roberta-base-go_emotions          (28 classes, GoEmotions)
    Audio: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  (VAD regression → mapped)

New models (v3 — this file):
    Text:  j-hartmann/emotion-english-distilroberta-base    (7 classes, Ekman + neutral)
    Audio: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition  (8 classes, direct)

Why the new models should be better:
    1. ALIGNED label spaces — both models use Ekman-based categories, so fusion
       is a direct label comparison instead of mapping 28 GoEmotions labels or
       converting continuous VAD dimensions to discrete labels.
    2. DIRECT classification — the audio model outputs categorical probabilities
       directly instead of VAD dimensions that require a hand-tuned mapping
       function (which is fragile and hard to validate).
    3. FEWER classes in text model — 7 instead of 28.  GoEmotions' fine-grained
       labels (admiration, amusement, approval, caring, curiosity, …) are
       interesting but cause confusion in customer-service contexts where the
       distinction between e.g. "annoyance" and "anger" rarely matters.
    4. DistilRoBERTa is ~40% faster than RoBERTa with comparable accuracy.

Other model suggestions worth evaluating:
    Text alternatives:
        • cardiffnlp/twitter-roberta-base-emotion-multilabel-latest
              — Trained on social/conversational text, good for informal speech
        • bhadresh-savani/distilbert-base-uncased-emotion
              — Lightweight, 6-class (Ekman), very fast inference
    Audio alternatives:
        • superb/wav2vec2-base-superb-er
              — SUPERB benchmark model, well-validated on IEMOCAP
        • facebook/hubert-large-ll60k fine-tuned on IEMOCAP
              — HuBERT often outperforms wav2vec2 on emotion tasks

All non-emotion logic (ASR, diarization, speaker-role classification, overlap
detection, fusion framework) is kept identical to main_v2.py.
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
from typing import Dict, List, Tuple
from collections import defaultdict

# ==============================================================================
# 0. COMPATIBILITY PATCHES  (identical to v2)
# ==============================================================================
print("Applying compatibility patches...")

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
print("[OK] torch.load patch applied")

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

try:
    import huggingface_hub
    _original_hf_hub_download = huggingface_hub.hf_hub_download
    def _patched_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return _original_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    print("[OK] huggingface_hub patch applied")
except ImportError:
    pass

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

DEFAULT_AUDIO_FILE = "Experiments/Voice-Generation/generated_audio/medium_overlap.mp3"

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Device: {DEVICE} ({COMPUTE_TYPE})")

warnings.filterwarnings("ignore")
if 'transformers' in sys.modules:
    transformers.logging.set_verbosity_error()

# ==============================================================================
# 2. EMOTION MODEL DEFINITIONS
# ==============================================================================

# --- V2 (current) models ---
V2_TEXT_EMOTION_MODEL   = "SamLowe/roberta-base-go_emotions"
V2_AUDIO_EMOTION_MODEL  = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# --- V3 (new) models ---
V3_TEXT_EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
V3_AUDIO_EMOTION_MODEL  = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Unified label set for v3 — both models naturally output from this set
# text:  anger, disgust, fear, joy, neutral, sadness, surprise
# audio: angry, calm, disgust, fearful, happy, neutral, sad, surprised
# We normalize both to a shared set:
UNIFIED_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Map each model's raw labels → unified set
TEXT_LABEL_MAP = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprise",
}

AUDIO_LABEL_MAP = {
    "angry":     "angry",
    "calm":      "neutral",   # calm ≈ neutral for customer-service context
    "disgust":   "disgust",
    "fearful":   "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprised": "surprise",
}

# ---------------------------------------------------------------------------
# Empathy / positive phrase sets (used in fusion correction)  — same as v2
# ---------------------------------------------------------------------------
EMPATHY_PHRASE_PATTERNS = [
    "understand your", "sorry to hear", "apologize for", "must be frustrating",
    "see why", "help you with", "figure this out", "resolve this", "i understand",
    "i can see how", "that must be", "i'm sorry about", "completely understand",
    "appreciate your patience", "take full responsibility"
]

POSITIVE_PHRASE_PATTERNS = [
    "thank you", "thanks", "appreciate", "grateful", "have a nice day", "great day",
    "take care", "good bye", "welcome", "my pleasure", "happy to help", "sounds good",
    "perfect", "great", "wonderful", "excellent", "awesome", "fantastic"
]

# ==============================================================================
# FUSION CONFIGURATION
# ==============================================================================
FUSION_CONFIG = {
    'text_confident_threshold':  0.70,
    'audio_confident_threshold': 0.75,
    'audio_override_threshold':  0.85,

    'text_weight':  0.65,
    'audio_weight': 0.35,

    'short_segment_threshold':   0.8,   # seconds
    'short_segment_text_weight': 0.85,
    'short_segment_audio_weight':0.15,

    'agent_audio_penalty': 0.4,
}


# ==============================================================================
# V2 AUDIO EMOTION — wav2vec2 regression head + VAD->label (same as main_v2)
# ==============================================================================

class RegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense    = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(transformers.Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config     = config
        self.wav2vec2   = transformers.Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        return {}

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        return pooled, self.classifier(pooled)


class V2AudioEmotionClassifier:
    """V2: Wav2Vec2 VAD regression → discrete label mapping."""

    def __init__(self):
        print(f"  Loading V2 Audio Emotion Model: {V2_AUDIO_EMOTION_MODEL}...")
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(V2_AUDIO_EMOTION_MODEL)
        self.model     = EmotionModel.from_pretrained(V2_AUDIO_EMOTION_MODEL)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        print("  [OK] V2 Audio Emotion Model Loaded")

    @staticmethod
    def _dimensions_to_emotion(arousal: float, valence: float, dominance: float) -> Tuple[str, float]:
        if arousal > 0.7:
            if valence > 0.5:
                return ("happy",   valence * 0.9 + arousal * 0.1)
            else:
                if dominance > 0.5:
                    return ("angry",   (1 - valence) * 0.8 + arousal * 0.2)
                else:
                    return ("fearful", (1 - valence) * 0.7 + arousal * 0.3)

        if valence > 0.6:
            if arousal >= 0.5:
                return ("happy",   valence)
            else:
                return ("neutral", valence)

        if valence < 0.4:
            if arousal > 0.6:
                if dominance > 0.5:
                    return ("angry",    1 - valence)
                else:
                    return ("fearful",  1 - valence)
            elif arousal < 0.4:
                return ("sad",      1 - valence)
            else:
                return ("disgusted", 1 - valence)

        return ("neutral", 0.5 + abs(valence - 0.5))

    def predict(self, audio_array: np.ndarray, sr: int = 16000) -> Dict:
        min_samples = sr
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

        inputs       = self.processor(audio_array, sampling_rate=sr)
        input_values = torch.tensor(inputs['input_values'][0]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, logits = self.model(input_values)
            dims      = logits[0].cpu().numpy()

        arousal, dominance, valence = float(dims[0]), float(dims[1]), float(dims[2])
        emotion, conf = self._dimensions_to_emotion(arousal, valence, dominance)

        return {
            'emotion':    emotion,
            'confidence': conf,
            'dimensions': {'arousal': arousal, 'dominance': dominance, 'valence': valence}
        }


# ==============================================================================
# V2 TEXT EMOTION — RoBERTa go_emotions (same as main_v2)
# ==============================================================================

class V2TextEmotionClassifier:
    def __init__(self):
        print(f"  Loading V2 Text Emotion Model: {V2_TEXT_EMOTION_MODEL}...")
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model=V2_TEXT_EMOTION_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None
        )
        print("  [OK] V2 Text Emotion Model Loaded")

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)',         r'$\1',   text)
        return " ".join(text.split())

    def predict(self, text: str) -> Dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        cleaned = [self._clean(t) if t.strip() else "" for t in texts]
        results = self.classifier(cleaned)

        out = []
        for i, res in enumerate(results):
            if not cleaned[i]:
                out.append({'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}})
                continue
            top        = max(res, key=lambda x: x['score'])
            all_scores = {r['label']: r['score'] for r in res}
            out.append({
                'emotion':    top['label'],
                'confidence': top['score'],
                'all_scores': all_scores
            })
        return out


# ==============================================================================
# V3 TEXT EMOTION — j-hartmann/emotion-english-distilroberta-base
# ==============================================================================
# 7 Ekman-based classes: anger, disgust, fear, joy, neutral, sadness, surprise
# Advantages over GoEmotions:
#   - Fewer, cleaner categories → less confusion
#   - Trained on 6 diverse emotion datasets → better generalization
#   - DistilRoBERTa backbone → ~40% faster inference
# ==============================================================================

class V3TextEmotionClassifier:
    """
    Categorical text emotion classifier using j-hartmann's DistilRoBERTa
    fine-tuned on multiple emotion datasets (Ekman 6 + neutral).
    """

    def __init__(self):
        print(f"  Loading V3 Text Emotion Model: {V3_TEXT_EMOTION_MODEL}...")
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model=V3_TEXT_EMOTION_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None
        )
        print("  [OK] V3 Text Emotion Model Loaded")

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)',         r'$\1',   text)
        return " ".join(text.split())

    def predict(self, text: str) -> Dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Batch prediction — single forward pass.
        Labels are normalized to the unified set via TEXT_LABEL_MAP.
        """
        cleaned = [self._clean(t) if t.strip() else "" for t in texts]
        results = self.classifier(cleaned)

        out = []
        for i, res in enumerate(results):
            if not cleaned[i]:
                out.append({'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}})
                continue

            # Normalize labels to unified set
            normalized = []
            for r in res:
                mapped = TEXT_LABEL_MAP.get(r['label'], r['label'])
                normalized.append({'label': mapped, 'score': r['score']})

            top        = max(normalized, key=lambda x: x['score'])
            all_scores = {r['label']: r['score'] for r in normalized}

            out.append({
                'emotion':    top['label'],
                'confidence': top['score'],
                'all_scores': all_scores
            })
        return out


# ==============================================================================
# V3 AUDIO EMOTION — ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
# ==============================================================================
# 8 direct classes:  angry, calm, disgust, fearful, happy, neutral, sad, surprised
# Advantages over VAD regression:
#   - DIRECT categorical output → no hand-tuned VAD→label mapping
#   - XLSR multilingual backbone → robust to accents
#   - Softmax probabilities → natural confidence scores
# ==============================================================================

class V3AudioEmotionClassifier:
    """
    Categorical audio emotion classifier using wav2vec2-lg-xlsr
    fine-tuned for speech emotion recognition.
    Outputs direct emotion categories with softmax probabilities.
    """

    def __init__(self):
        print(f"  Loading V3 Audio Emotion Model: {V3_AUDIO_EMOTION_MODEL}...")
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(V3_AUDIO_EMOTION_MODEL)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(V3_AUDIO_EMOTION_MODEL)

        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

        # Build label list from model config
        self.labels = [self.model.config.id2label[i]
                       for i in range(self.model.config.num_labels)]

        print(f"  [OK] V3 Audio Emotion Model Loaded (labels: {self.labels})")

    def predict(self, audio_array: np.ndarray, sr: int = 16000) -> Dict:
        """
        Predict emotion from raw audio waveform.
        Returns unified label, confidence, and full score distribution.
        """
        min_samples = sr  # 1 second minimum
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(DEVICE)
        if "attention_mask" in inputs:
            attention_mask = inputs.attention_mask.to(DEVICE)
        else:
            attention_mask = None

        with torch.no_grad():
            if attention_mask is not None:
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits

            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

        # Build per-label scores and get top prediction
        raw_scores = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        # Normalize labels to unified set
        unified_scores = {}
        for raw_label, score in raw_scores.items():
            mapped = AUDIO_LABEL_MAP.get(raw_label, raw_label)
            # If two raw labels map to the same unified label (e.g. calm+neutral → neutral),
            # take the max score since they represent the same concept
            if mapped in unified_scores:
                unified_scores[mapped] = max(unified_scores[mapped], score)
            else:
                unified_scores[mapped] = score

        top_label = max(unified_scores, key=unified_scores.get)
        top_conf  = unified_scores[top_label]

        return {
            'emotion':    top_label,
            'confidence': top_conf,
            'all_scores': unified_scores,
            'raw_scores': raw_scores,     # keep raw for debugging
        }


# ==============================================================================
# EMOTION FUSION  (shared by both v2 and v3)
# ==============================================================================

def apply_fusion_logic(text_res: Dict, audio_res: Dict, text_content: str,
                       segment_duration: float = 1.0,
                       speaker_role: str = 'UNKNOWN') -> Dict:
    """
    Combine text and audio emotion predictions.
    Returns actual emotion labels (not coarse sentiment).
    """
    cfg = FUSION_CONFIG
    text_lower = text_content.lower()

    t_emotion, t_conf = text_res['emotion'], text_res['confidence']
    a_emotion, a_conf = audio_res['emotion'], audio_res['confidence']

    original_emotion = t_emotion
    was_corrected    = False

    # ------------------------------------------------------------------
    # Rule-based text corrections (empathy / politeness misclassified)
    # ------------------------------------------------------------------
    negative_emotions = {'angry', 'anger', 'sadness', 'sad', 'disgust', 'fear',
                         'fearful', 'annoyance', 'disapproval'}

    if t_emotion in negative_emotions:
        for phrase in EMPATHY_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion     = 'neutral'
                t_conf        = max(t_conf, 0.85)
                was_corrected = True
                break

    if t_emotion in negative_emotions and not was_corrected:
        for phrase in POSITIVE_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion     = 'neutral'
                t_conf        = max(t_conf, 0.75)
                was_corrected = True
                break

    # ------------------------------------------------------------------
    # Weight calculation
    # ------------------------------------------------------------------
    text_weight  = cfg['text_weight']
    audio_weight = cfg['audio_weight']
    is_short     = segment_duration < cfg['short_segment_threshold']
    is_agent     = speaker_role == 'AGENT'

    if is_short:
        text_weight  = cfg['short_segment_text_weight']
        audio_weight = cfg['short_segment_audio_weight']

    if is_agent:
        audio_weight *= cfg['agent_audio_penalty']

    a_conf_eff = a_conf * 0.6 if is_short else a_conf

    # ------------------------------------------------------------------
    # Fusion decision
    # ------------------------------------------------------------------
    if t_conf >= cfg['text_confident_threshold']:
        fused_emotion, fused_conf, source = t_emotion, t_conf, "TEXT >=70%"

    elif (not is_agent and a_conf_eff >= cfg['audio_confident_threshold']) or \
         (is_agent      and a_conf     >= cfg['audio_override_threshold']):
        fused_emotion, fused_conf, source = a_emotion, a_conf, "AUDIO >=75%"

    elif t_emotion == a_emotion:
        fused_emotion = t_emotion
        fused_conf    = min(1.0, (t_conf + a_conf) / 1.5)
        source        = "BOTH AGREE"

    else:
        w_text  = t_conf * text_weight
        w_audio = a_conf_eff * audio_weight

        if w_text >= w_audio:
            fused_emotion, fused_conf, source = t_emotion, t_conf, "TEXT weighted"
        else:
            fused_emotion, fused_conf, source = a_emotion, a_conf, "AUDIO weighted"

        if t_conf < 0.50 and a_conf_eff < 0.50:
            fused_emotion = 'neutral'
            fused_conf    = 0.40
            source        = "LOW CONF"

    return {
        'emotion':          fused_emotion,
        'confidence':       fused_conf,
        'source':           source,
        'corrected':        was_corrected,
        'original_emotion': original_emotion if was_corrected else None,
        'text_details': {
            'emotion':    t_emotion,
            'confidence': t_conf,
            'all_scores': text_res.get('all_scores', {})
        },
        'audio_details': {
            'emotion':    a_emotion,
            'confidence': a_conf,
            'dimensions': audio_res.get('dimensions', {}),
            'all_scores': audio_res.get('all_scores', {}),
        }
    }


# ==============================================================================
# SPEAKER ROLE CLASSIFIER  (identical to v2)
# ==============================================================================

class ProductionSpeakerRoleClassifier:
    """
    4-component ensemble for speaker-role classification:
        1. Zero-shot LLM  (BART-large-MNLI)
        2. Conversation-structure analysis
        3. Linguistic-pattern matching
        4. Relative turn-taking dynamics
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("   Initializing Production Speaker Role Classifier...")

        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        print("   [1/4] Loading zero-shot LLM (BART-large-MNLI)...")
        zs_name      = "facebook/bart-large-mnli"
        zs_tokenizer = AutoTokenizer.from_pretrained(zs_name)
        zs_model     = AutoModelForSequenceClassification.from_pretrained(zs_name, use_safetensors=True)
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=zs_model,
            tokenizer=zs_tokenizer,
            device=0 if device == "cuda" else -1
        )

        print("   [2/4] Initializing conversation-structure analyzer...")
        self.greeting_patterns = [
            r"^(hello|hi|hey|good\s+(morning|afternoon|evening))",
            r"(thank you for (calling|contacting|reaching out))",
            r"(you('ve)?\s+reached|this is\s+\w+\s+support)",
            r"(how\s+(can|may)\s+(i|we)\s+(help|assist))",
            r"(welcome to)",
            r"(thank you for (calling|contacting)\s+)",
            r"(this is\s+\w+\s+from)",
            r"(i'll be (helping|assisting) you)",
        ]
        self.closing_patterns = [
            "anything else", "is there anything", "have a great day",
            "have a wonderful day", "thank you for calling",
            "thank you for contacting", "take care", "goodbye",
            "i hope that helps", "is there anything else i can do",
            "glad i could help", "hope we got that sorted",
        ]

        print("   [3/4] Initializing industry-grade linguistic patterns...")
        self.agent_indicators = {
            'strong': [
                "i can see your account", "looking at your account",
                "pulling up your account", "i'm seeing here",
                "on my end", "system shows", "i have your account",
                "account is showing", "how can i help", "how may i help",
                "thank you for calling", "thank you for contacting",
                "you've reached", "this is tech support",
                "my name is", "i'll be assisting",
                "let me transfer", "i'll escalate",
                "ticket number", "case number",
                "reference number", "i'll create a ticket",
                "can i verify", "for security purposes",
                "can you confirm your", "verify your identity",
                "date of birth please", "account holder",
                "here's what we can do", "the steps are",
                "you should see", "that should fix",
                "let me walk you through", "i'd like to help you",
                "there is currently", "we are aware",
                "known issue", "our team is working",
                "estimated time", "scheduled maintenance",
            ],
            'moderate': [
                "let me check", "i understand your",
                "i apologize for", "let me help you",
                "i'll assist you", "happy to help",
                "according to", "shows that",
                "i can see", "on our side",
                "let me look into", "i'll look into that",
                "bear with me", "one moment please",
                "i appreciate your", "bear with me while",
                "just a moment", "please hold",
                "i'll get that sorted", "let me arrange",
            ],
        }
        self.customer_indicators = {
            'strong': [
                "my phone", "my account", "my bill", "my order",
                "my internet", "my service", "my subscription", "my card",
                "i have a problem", "i need help",
                "keeps crashing", "not working",
                "i'm having trouble", "it stopped working",
                "it's not working", "it won't",
                "doesn't work", "broke down",
                "this is ridiculous", "this is unacceptable",
                "i've been waiting", "i want to speak to",
                "i want to cancel", "cancel my",
                "i want a refund", "give me a refund",
                "i was charged", "charged me",
                "i didn't receive", "never received",
                "under warranty", "still under warranty",
                "i paid for", "i'm paying for",
            ],
            'moderate': [
                "can you help", "why is", "how do i", "i've tried",
                "i'm frustrated", "i'm upset", "nobody helps",
                "this keeps happening", "again?", "i already told",
                "i called before", "last time", "still not fixed",
                "same problem", "what's going on", "what happened to",
            ],
        }

        print("   [4/4] Configuring ensemble weights...")
        self.solution_phrases = [
            "try restarting", "clear the cache", "go to settings", "click on",
            "you need to", "you should", "the steps are", "first,",
            "next,", "then,", "make sure", "uninstall",
            "reinstall", "update", "download", "enable",
            "disable", "change your", "reset your", "restart your",
            "check your", "navigate to", "select", "tap on",
            "press", "enter your",
        ]
        self.ensemble_weights = {
            'zero_shot':             0.35,
            'conversation_structure':0.25,
            'linguistic_patterns':   0.23,
            'turn_taking':           0.17,
        }
        print("   [OK] Production classifier ready\n")

    def classify_with_zero_shot(self, text: str) -> Dict[str, float]:
        labels = [
            "a customer support representative",
            "a customer seeking help"
        ]
        try:
            result = self.zero_shot_classifier(text[:1000], labels, multi_label=False)
            agent_prob = result['scores'][0] if result['labels'][0] == labels[0] else result['scores'][1]
            return {'agent_score': agent_prob, 'confidence': max(result['scores'])}
        except Exception as e:
            print(f"      Warning: Zero-shot failed: {e}")
            return {'agent_score': 0.5, 'confidence': 0.0}

    def analyze_conversation_structure(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        all_speakers = [s.get('speaker') for s in segments if s.get('speaker')]
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]

        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        agent_score = 0.5
        confidence  = 0.0

        if all_speakers and speaker_id == all_speakers[0]:
            first_text = speaker_segs[0].get('text', '').lower()
            has_greeting = any(re.search(p, first_text) for p in self.greeting_patterns)
            complaint_start = any(kw in first_text for kw in [
                "my phone", "my account", "not working", "i need help",
                "i have a problem", "keeps crashing", "i'm having trouble",
                "i want to", "this is ridiculous", "i called"
            ])

            if has_greeting and not complaint_start:
                agent_score += 0.40
                confidence   = 0.95
            elif complaint_start:
                agent_score -= 0.30
                confidence   = 0.85
            else:
                agent_score += 0.20
                confidence   = 0.65

        first_three = " ".join(s.get('text', '') for s in speaker_segs[:3]).lower()
        if any(re.search(p, first_three) for p in self.greeting_patterns):
            agent_score += 0.15
            confidence   = max(confidence, 0.80)

        if len(speaker_segs) > 1:
            last_two = " ".join(s.get('text', '') for s in speaker_segs[-2:]).lower()
            if any(cp in last_two for cp in self.closing_patterns):
                agent_score += 0.15
                confidence   = max(confidence, 0.85)

        agent_score = max(0.0, min(1.0, agent_score))
        return {'agent_score': agent_score, 'confidence': confidence}

    def analyze_linguistic_patterns(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()

        strong_a   = sum(1 for p in self.agent_indicators['strong']   if p in text_lower)
        moderate_a = sum(1 for p in self.agent_indicators['moderate'] if p in text_lower)
        agent_raw  = strong_a * 0.40 + moderate_a * 0.18

        strong_c   = sum(1 for p in self.customer_indicators['strong']   if p in text_lower)
        moderate_c = sum(1 for p in self.customer_indicators['moderate'] if p in text_lower)
        cust_raw   = strong_c * 0.40 + moderate_c * 0.18

        agent_score = max(0.0, min(1.0, 0.5 + agent_raw - cust_raw))

        if strong_a > 0 or strong_c > 0:
            conf = 0.92
        elif moderate_a > 0 or moderate_c > 0:
            conf = 0.72
        else:
            conf = 0.30

        return {'agent_score': agent_score, 'confidence': conf}

    def analyze_turn_taking(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        other_segs   = [s for s in segments if s.get('speaker') != speaker_id and s.get('speaker')]

        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        my_words    = np.mean([len(s.get('text', '').split()) for s in speaker_segs])
        other_words = np.mean([len(s.get('text', '').split()) for s in other_segs]) if other_segs else my_words
        ratio       = my_words / other_words if other_words > 0 else 1.0

        if ratio > 1.3:
            word_score, word_conf = 0.70, 0.65
        elif ratio < 0.75:
            word_score, word_conf = 0.30, 0.65
        else:
            word_score, word_conf = 0.50, 0.35

        my_text    = " ".join(s.get('text', '') for s in speaker_segs)
        other_text = " ".join(s.get('text', '') for s in other_segs) if other_segs else ""

        my_q    = my_text.count('?')   / max(len(speaker_segs), 1)
        other_q = other_text.count('?')/ max(len(other_segs), 1) if other_segs else 0.0

        if my_q > other_q + 0.15:
            q_score, q_conf = 0.65, 0.60
        elif my_q < other_q - 0.15:
            q_score, q_conf = 0.35, 0.60
        else:
            q_score, q_conf = 0.50, 0.30

        my_solutions = sum(1 for p in self.solution_phrases if p in my_text.lower())
        if my_solutions >= 2:
            sol_score, sol_conf = 0.85, 0.80
        elif my_solutions == 1:
            sol_score, sol_conf = 0.70, 0.55
        else:
            sol_score, sol_conf = 0.50, 0.25

        agent_score = (word_score * 0.30 + q_score * 0.25 + sol_score * 0.45)
        confidence  = (word_conf  * 0.30 + q_conf  * 0.25 + sol_conf  * 0.45)
        agent_score = max(0.0, min(1.0, agent_score))
        return {'agent_score': agent_score, 'confidence': confidence}

    def ensemble_classification(self, zero_shot, structure, linguistic, turn_taking):
        components = {
            'zero_shot':              zero_shot,
            'conversation_structure': structure,
            'linguistic_patterns':    linguistic,
            'turn_taking':            turn_taking,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        debug_info   = {}

        for name, comp in components.items():
            base_w = self.ensemble_weights[name]
            score  = comp['agent_score']
            conf   = comp['confidence']
            eff_w  = base_w * (0.5 + conf * 0.5)

            weighted_sum += score * eff_w
            total_weight += eff_w
            debug_info[name] = {
                'score': score, 'confidence': conf, 'effective_weight': eff_w
            }

        agent_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

        if agent_prob > 0.55:
            role, final_conf = "AGENT", agent_prob
        elif agent_prob < 0.45:
            role, final_conf = "CUSTOMER", 1 - agent_prob
        else:
            if structure['agent_score'] > 0.55:
                role, final_conf = "AGENT", 0.68
            else:
                role, final_conf = "CUSTOMER", 0.68

        debug_info['final'] = {
            'agent_probability': agent_prob, 'role': role, 'final_confidence': final_conf,
        }
        return role, final_conf, debug_info

    def classify_speaker_role(self, speaker_id: str, segments: List[Dict]):
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        if not speaker_segs:
            return "UNKNOWN", 0.0, {}
        full_text = " ".join(s.get('text', '') for s in speaker_segs)
        zs   = self.classify_with_zero_shot(full_text)
        st   = self.analyze_conversation_structure(speaker_id, segments)
        ling = self.analyze_linguistic_patterns(full_text)
        tt   = self.analyze_turn_taking(speaker_id, segments)
        return self.ensemble_classification(zs, st, ling, tt)

    def classify_all_speakers(self, segments: List[Dict]) -> Dict[str, str]:
        speakers = list({s.get('speaker') for s in segments if s.get('speaker')})
        results  = {}

        print("   Classifying speaker roles (Production Ensemble)...")
        print("   " + "=" * 60)

        for spk in speakers:
            role, conf, debug = self.classify_speaker_role(spk, segments)
            results[spk] = role
            print(f"\n   {spk}: {role} (confidence: {conf:.2f})")
            print(f"   Component breakdown:")
            for comp_name, info in debug.items():
                if comp_name != 'final':
                    print(f"      {comp_name:25s}: score={info['score']:.2f}, "
                          f"conf={info['confidence']:.2f}, weight={info['effective_weight']:.3f}")

        print("   " + "=" * 60)

        if len(results) == 2:
            speakers_list = list(results.keys())
            roles = [results[s] for s in speakers_list]
            if roles[0] == roles[1]:
                all_spk       = [s.get('speaker') for s in segments if s.get('speaker')]
                first_speaker = all_spk[0] if all_spk else speakers_list[0]
                print(f"\n   >> Both classified as {roles[0]}. "
                      f"Applying first-speaker heuristic: {first_speaker} -> AGENT")
                results[first_speaker] = 'AGENT'
                other = [s for s in speakers_list if s != first_speaker][0]
                results[other] = 'CUSTOMER'

        return results


# ==============================================================================
# OVERLAP DETECTION  (identical to v2)
# ==============================================================================

def detect_overlaps(segments: List[Dict], threshold: float = 0.1) -> List[Dict]:
    segments = sorted(segments, key=lambda x: x['start'])
    for seg in segments:
        seg.setdefault('overlap', False)

    for i, curr in enumerate(segments):
        for j in range(i + 1, len(segments)):
            nxt = segments[j]
            if nxt['start'] >= curr['end']:
                break
            curr['overlap'] = True
            nxt['overlap']  = True
    return segments


# ==============================================================================
# COMPARISON PIPELINE — runs both v2 and v3 emotion models side-by-side
# ==============================================================================

class ComparisonPipeline:
    """
    Runs ASR + diarization + role classification once, then applies BOTH
    v2 and v3 emotion models to every segment for side-by-side comparison.
    """

    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing Emotion Comparison Pipeline ===")

    # ------------------------------------------------------------------
    def _load_whisper_and_diarize(self):
        print(">> Loading ASR & Diarization Models...")
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        asr_model     = whisperx.load_model(self.whisper_model_size, DEVICE, compute_type=COMPUTE_TYPE)
        print("     [OK] WhisperX Loaded")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("     [OK] Diarization Loaded")
        return whisperx, asr_model, diarize_model

    # ------------------------------------------------------------------
    def _load_analysis_models(self):
        """Load ALL emotion models (v2 + v3) plus role classifier."""
        print(">> Loading Analysis Models...")
        return {
            'v2_text':  V2TextEmotionClassifier(),
            'v2_audio': V2AudioEmotionClassifier(),
            'v3_text':  V3TextEmotionClassifier(),
            'v3_audio': V3AudioEmotionClassifier(),
            'role':     ProductionSpeakerRoleClassifier(device=DEVICE),
        }

    # ------------------------------------------------------------------
    def process(self, audio_path: str, language: str = None) -> Dict:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return {}

        print(f"Processing: {audio_path}")
        start_time = time.time()

        # ==========================================================
        # PHASE 1 — ASR + DIARIZATION  (shared, run once)
        # ==========================================================
        whisperx, asr_model, diarize_model = self._load_whisper_and_diarize()

        print(">> Step 1: Transcribing...")
        audio  = whisperx.load_audio(audio_path)
        result = asr_model.transcribe(audio, batch_size=16, language=language)
        language = result["language"]
        print(f"   Language: {language}")

        print(">> Step 2: Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE,
                                return_char_alignments=False)

        print(">> Step 3: Diarizing...")
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        print(">> Step 4: Detecting Overlaps...")
        result["segments"] = detect_overlaps(result["segments"])

        # Cleanup Phase 1
        print(">> Cleaning up ASR models...")
        del asr_model, diarize_model, model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Memory Freed")

        # ==========================================================
        # PHASE 2 — ROLE CLASSIFICATION  (shared, run once)
        # ==========================================================
        print(">> Step 5: Speaker Role Classification...")
        models   = self._load_analysis_models()
        role_clf = models['role']

        flat_segments = [
            {"text": seg["text"].strip(), "speaker": seg.get("speaker", "UNKNOWN")}
            for seg in result["segments"]
        ]
        speaker_roles = role_clf.classify_all_speakers(flat_segments)

        del role_clf, models['role']
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Role classifier freed")

        # ==========================================================
        # PHASE 3 — EMOTION COMPARISON
        # ==========================================================
        print("\n>> Step 6: Running Emotion Comparison (V2 vs V3)...")
        print("=" * 80)

        v2_text_clf  = models['v2_text']
        v2_audio_clf = models['v2_audio']
        v3_text_clf  = models['v3_text']
        v3_audio_clf = models['v3_audio']

        # Batch text predictions for both
        all_texts         = [seg["text"].strip() for seg in result["segments"]]
        v2_text_preds     = v2_text_clf.predict_batch(all_texts)
        v3_text_preds     = v3_text_clf.predict_batch(all_texts)

        # Load audio once
        full_audio, _ = librosa.load(audio_path, sr=16000)

        comparison_segments = []
        agreement_stats = {'text_agree': 0, 'audio_agree': 0, 'fusion_agree': 0, 'total': 0}

        for idx, seg in enumerate(result["segments"]):
            start   = seg["start"]
            end     = seg["end"]
            text    = seg["text"].strip()
            speaker = seg.get("speaker", "UNKNOWN")
            role    = speaker_roles.get(speaker, "UNKNOWN")
            duration = end - start

            # Audio slice
            audio_seg = full_audio[int(start * 16000):int(end * 16000)]

            # V2 predictions
            v2_text_res  = v2_text_preds[idx]
            v2_audio_res = v2_audio_clf.predict(audio_seg)
            v2_fusion    = apply_fusion_logic(v2_text_res, v2_audio_res, text,
                                              segment_duration=duration,
                                              speaker_role=role)

            # V3 predictions
            v3_text_res  = v3_text_preds[idx]
            v3_audio_res = v3_audio_clf.predict(audio_seg)
            v3_fusion    = apply_fusion_logic(v3_text_res, v3_audio_res, text,
                                              segment_duration=duration,
                                              speaker_role=role)

            # Track agreement
            agreement_stats['total'] += 1
            if v2_text_res['emotion'] == v3_text_res['emotion']:
                agreement_stats['text_agree'] += 1
            if v2_audio_res['emotion'] == v3_audio_res['emotion']:
                agreement_stats['audio_agree'] += 1
            if v2_fusion['emotion'] == v3_fusion['emotion']:
                agreement_stats['fusion_agree'] += 1

            seg_result = {
                "start":   start,
                "end":     end,
                "speaker": speaker,
                "role":    role,
                "text":    text,
                "overlap": seg.get('overlap', False),
                "v2": {
                    "text_emotion":   v2_text_res['emotion'],
                    "text_conf":      v2_text_res['confidence'],
                    "audio_emotion":  v2_audio_res['emotion'],
                    "audio_conf":     v2_audio_res['confidence'],
                    "audio_dims":     v2_audio_res.get('dimensions', {}),
                    "fused_emotion":  v2_fusion['emotion'],
                    "fused_conf":     v2_fusion['confidence'],
                    "fused_source":   v2_fusion['source'],
                    "corrected":      v2_fusion.get('corrected', False),
                },
                "v3": {
                    "text_emotion":   v3_text_res['emotion'],
                    "text_conf":      v3_text_res['confidence'],
                    "audio_emotion":  v3_audio_res['emotion'],
                    "audio_conf":     v3_audio_res['confidence'],
                    "audio_scores":   v3_audio_res.get('all_scores', {}),
                    "fused_emotion":  v3_fusion['emotion'],
                    "fused_conf":     v3_fusion['confidence'],
                    "fused_source":   v3_fusion['source'],
                    "corrected":      v3_fusion.get('corrected', False),
                },
            }
            comparison_segments.append(seg_result)

            # Pretty-print comparison
            self._print_comparison(seg_result)

        # ==========================================================
        # SUMMARY
        # ==========================================================
        elapsed = time.time() - start_time
        total   = agreement_stats['total']

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Total segments analyzed: {total}")
        print(f"Text emotion agreement:  {agreement_stats['text_agree']}/{total} "
              f"({agreement_stats['text_agree']/total*100:.1f}%)")
        print(f"Audio emotion agreement: {agreement_stats['audio_agree']}/{total} "
              f"({agreement_stats['audio_agree']/total*100:.1f}%)")
        print(f"Fused emotion agreement: {agreement_stats['fusion_agree']}/{total} "
              f"({agreement_stats['fusion_agree']/total*100:.1f}%)")
        print(f"\nProcessing time: {elapsed:.1f}s")

        # Compute confidence stats
        v2_text_confs  = [s['v2']['text_conf']  for s in comparison_segments]
        v3_text_confs  = [s['v3']['text_conf']  for s in comparison_segments]
        v2_audio_confs = [s['v2']['audio_conf'] for s in comparison_segments]
        v3_audio_confs = [s['v3']['audio_conf'] for s in comparison_segments]
        v2_fused_confs = [s['v2']['fused_conf'] for s in comparison_segments]
        v3_fused_confs = [s['v3']['fused_conf'] for s in comparison_segments]

        print(f"\nAvg Text Confidence:   V2={np.mean(v2_text_confs):.3f}  "
              f"V3={np.mean(v3_text_confs):.3f}")
        print(f"Avg Audio Confidence:  V2={np.mean(v2_audio_confs):.3f}  "
              f"V3={np.mean(v3_audio_confs):.3f}")
        print(f"Avg Fused Confidence:  V2={np.mean(v2_fused_confs):.3f}  "
              f"V3={np.mean(v3_fused_confs):.3f}")

        # Disagreement analysis
        disagreements = [s for s in comparison_segments
                         if s['v2']['fused_emotion'] != s['v3']['fused_emotion']]
        if disagreements:
            print(f"\n--- Disagreements ({len(disagreements)} segments) ---")
            for d in disagreements:
                print(f"  [{d['start']:.1f}s-{d['end']:.1f}s] {d['role']}: "
                      f"\"{d['text'][:60]}...\"")
                print(f"    V2: {d['v2']['fused_emotion']:>10s} ({d['v2']['fused_conf']:.1%}) "
                      f"  V3: {d['v3']['fused_emotion']:>10s} ({d['v3']['fused_conf']:.1%})")

        return {
            'segments':    comparison_segments,
            'agreement':   agreement_stats,
            'elapsed':     elapsed,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _print_comparison(seg: Dict):
        v2, v3 = seg['v2'], seg['v3']
        role   = seg['role']
        overlap_tag = " [OVERLAP]" if seg.get('overlap') else ""

        # Highlight when models disagree
        text_match  = "==" if v2['text_emotion']  == v3['text_emotion']  else "!="
        audio_match = "==" if v2['audio_emotion'] == v3['audio_emotion'] else "!="
        fused_match = "==" if v2['fused_emotion'] == v3['fused_emotion'] else "!="

        marker = "  DISAGREE >>>" if fused_match == "!=" else ""

        print(f"\n[{seg['start']:.1f}s-{seg['end']:.1f}s] {role}{overlap_tag}{marker}")
        print(f"   Text: \"{seg['text'][:70]}{'...' if len(seg['text']) > 70 else ''}\"")
        print(f"   Text Emotion:   V2={v2['text_emotion']:>12s} ({v2['text_conf']:5.1%})  "
              f"{text_match}  V3={v3['text_emotion']:>10s} ({v3['text_conf']:5.1%})")
        print(f"   Audio Emotion:  V2={v2['audio_emotion']:>12s} ({v2['audio_conf']:5.1%})  "
              f"{audio_match}  V3={v3['audio_emotion']:>10s} ({v3['audio_conf']:5.1%})")
        print(f"   >> Fused:       V2={v2['fused_emotion']:>12s} ({v2['fused_conf']:5.1%})  "
              f"{fused_match}  V3={v3['fused_emotion']:>10s} ({v3['fused_conf']:5.1%})")


# ==============================================================================
# V3 STANDALONE PIPELINE  (drop-in replacement for main_v2's CombinedPipeline)
# ==============================================================================

class V3Pipeline:
    """
    Production pipeline using the new v3 emotion models only.
    Identical structure to CombinedPipeline in main_v2.py but with:
        Text:  j-hartmann/emotion-english-distilroberta-base
        Audio: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    """

    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing V3 Production Pipeline ===")

    def _load_whisper_and_diarize(self):
        print(">> Loading ASR & Diarization Models...")
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        asr_model     = whisperx.load_model(self.whisper_model_size, DEVICE, compute_type=COMPUTE_TYPE)
        print("     [OK] WhisperX Loaded")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("     [OK] Diarization Loaded")
        return whisperx, asr_model, diarize_model

    def _load_analysis_models(self):
        print(">> Loading V3 Analysis Models (Emotion & Role)...")
        return {
            'text':  V3TextEmotionClassifier(),
            'audio': V3AudioEmotionClassifier(),
            'role':  ProductionSpeakerRoleClassifier(device=DEVICE),
        }

    def process(self, audio_path: str, language: str = None) -> List[Dict]:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return []

        print(f"Processing: {audio_path}")
        start_time = time.time()

        # PHASE 1 — ASR + DIARIZATION
        whisperx, asr_model, diarize_model = self._load_whisper_and_diarize()

        print(">> Step 1: Transcribing...")
        audio  = whisperx.load_audio(audio_path)
        result = asr_model.transcribe(audio, batch_size=16, language=language)
        language = result["language"]
        print(f"   Language: {language}")

        print(">> Step 2: Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE,
                                return_char_alignments=False)

        print(">> Step 3: Diarizing...")
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        print(">> Step 4: Detecting Overlaps...")
        result["segments"] = detect_overlaps(result["segments"])

        print(">> Cleaning up ASR models...")
        del asr_model, diarize_model, model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Memory Freed")

        # PHASE 2 — ROLE + EMOTION
        print(">> Step 5: Speaker Role Classification...")
        models   = self._load_analysis_models()
        role_clf = models['role']
        text_clf = models['text']
        audio_clf = models['audio']

        flat_segments = [
            {"text": seg["text"].strip(), "speaker": seg.get("speaker", "UNKNOWN")}
            for seg in result["segments"]
        ]
        speaker_roles = role_clf.classify_all_speakers(flat_segments)

        del role_clf, models['role']
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Role classifier freed")

        # PHASE 3 — EMOTION (batched text + sequential audio)
        print("\n>> Step 6: Analyzing Emotions (V3 models)...")

        all_texts        = [seg["text"].strip() for seg in result["segments"]]
        text_predictions = text_clf.predict_batch(all_texts)

        full_audio, _ = librosa.load(audio_path, sr=16000)

        final_segments = []
        for idx, seg in enumerate(result["segments"]):
            start    = seg["start"]
            end      = seg["end"]
            text     = seg["text"].strip()
            speaker  = seg.get("speaker", "UNKNOWN")
            role     = speaker_roles.get(speaker, "UNKNOWN")

            audio_seg = full_audio[int(start * 16000):int(end * 16000)]
            audio_res = audio_clf.predict(audio_seg)
            text_res  = text_predictions[idx]

            fusion = apply_fusion_logic(text_res, audio_res, text,
                                        segment_duration=end - start,
                                        speaker_role=role)

            final_segments.append({
                "start":            start,
                "end":              end,
                "speaker":          speaker,
                "role":             role,
                "text":             text,
                "emotion_analysis": fusion,
            })

            self._print_utterance(speaker, role, text, start, end,
                                  text_res, audio_res, fusion,
                                  is_overlap=seg.get('overlap', False))

        elapsed = time.time() - start_time
        print(f"\n>> Total processing time: {elapsed:.1f}s")
        return final_segments

    @staticmethod
    def _print_utterance(speaker, role, text, start, end, text_res, audio_res, fusion, is_overlap=False):
        overlap_tag = " [OVERLAP]" if is_overlap else ""

        t_emo, t_conf = text_res['emotion'],  text_res['confidence']
        a_emo, a_conf = audio_res['emotion'], audio_res['confidence']

        f_emo, f_conf, f_src = fusion['emotion'], fusion['confidence'], fusion['source']
        corrected_str = (f" (was: {fusion['original_emotion']})"
                         if fusion.get('corrected') and fusion.get('original_emotion') else "")

        print(f"\n[{start:.1f}s-{end:.1f}s] {role}{overlap_tag}")
        print(f"   Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"   Text Emotion:  {t_emo:12s} ({t_conf:5.1%}){corrected_str}")
        print(f"   Audio Emotion: {a_emo:12s} ({a_conf:5.1%})")
        print(f"   >> Fused:      {f_emo:12s} ({f_conf:5.1%}) [{f_src}]")


# ==============================================================================
# JSON SERIALIZATION HELPER
# ==============================================================================

def save_results_json(results, output_path: str):
    def _convert(obj):
        if isinstance(obj, dict):            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):            return [_convert(v) for v in obj]
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(_convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VocalMind Emotion Model Comparison (V2 vs V3)")
    parser.add_argument("--audio",    type=str, default=DEFAULT_AUDIO_FILE,
                        help="Input audio file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output",   type=str,
                        help="Save comparison results to JSON")
    parser.add_argument("--mode",     type=str, default="compare",
                        choices=["compare", "v3only"],
                        help="'compare' = run both v2 & v3 side-by-side, "
                             "'v3only' = run v3 pipeline only (drop-in replacement)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        sys.exit(1)

    if args.mode == "compare":
        pipeline = ComparisonPipeline()
        results  = pipeline.process(args.audio, language=args.language)
    else:
        pipeline = V3Pipeline()
        results  = pipeline.process(args.audio, language=args.language)

    if args.output and results:
        save_results_json(results, args.output)

    print("\n" + "=" * 50)
    print("DONE.")
