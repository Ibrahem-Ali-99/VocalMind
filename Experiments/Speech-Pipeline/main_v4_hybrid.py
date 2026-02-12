"""
VocalMind Pipeline v4 — Hybrid Emotion Model (Best-of-Both)
============================================================

Combines:
    Text:  j-hartmann/emotion-english-distilroberta-base  (7 Ekman classes)
    Audio: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  (VAD regression)

Why this combination:
    - j-hartmann text model gives cleaner Ekman labels (7 vs 28 GoEmotions)
      and better detects surprise, anger on short phrases.
    - audeering audio model actually produces confident, meaningful signals
      (unlike ehcalabres which output near-uniform ~14% per class).
    - Fix: j-hartmann over-predicts "sadness" for frustrated/demanding
      utterances in CS context. We add a context-aware correction layer
      that remaps sadness→angry when frustration indicators are present.

Compared to pure V2:
    - Cleaner label taxonomy (no "admiration" vs "approval" vs "caring")
    - Better surprise/anger detection on short phrases
    - Sadness correction prevents mis-routing frustrated customers

Run modes:
    --mode compare   : V2 (pure) vs V4 (hybrid) side-by-side
    --mode v4only    : Drop-in replacement for main_v2 pipeline
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
# 0. COMPATIBILITY PATCHES
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
# 2. MODEL DEFINITIONS
# ==============================================================================

# V2 models (baseline)
V2_TEXT_EMOTION_MODEL  = "SamLowe/roberta-base-go_emotions"
V2_AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# V4 hybrid: j-hartmann text + audeering audio (same audio as V2)
V4_TEXT_EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"
V4_AUDIO_EMOTION_MODEL = V2_AUDIO_EMOTION_MODEL  # intentionally reuse — it works

# j-hartmann raw labels → unified labels
TEXT_LABEL_MAP = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprise",
}

# ---------------------------------------------------------------------------
# Phrase sets for fusion correction (same as v2)
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

# ---------------------------------------------------------------------------
# V4 SADNESS CORRECTION — customer-service context-aware remap
# ---------------------------------------------------------------------------
# j-hartmann over-predicts "sad" for utterances that express frustration,
# demands, or complaints. In CS context, these are anger/frustration signals
# that need escalation, not empathy-for-grief responses.
#
# Strategy:
#   If text model says "sad" AND frustration indicators are present → "angry"
#   If text model says "sad" AND empathy phrases are present → "neutral"
#     (agent expressing empathy, not actual sadness)
# ---------------------------------------------------------------------------

FRUSTRATION_INDICATORS = [
    # Direct demands
    "i want", "i need", "i demand", "give me", "i expect",
    # Complaints
    "charged", "overcharged", "not working", "doesn't work", "won't work",
    "broken", "failed", "this is", "unacceptable", "ridiculous",
    "terrible", "horrible", "worst", "never", "can't believe",
    # Urgency
    "immediately", "right now", "asap", "hurry", "urgent",
    # Threats
    "cancel", "refund", "lawsuit", "lawyer", "report", "complaint",
    "speak to", "manager", "supervisor",
    # Repeated issues
    "again", "still", "already", "how many times", "keep",
    "been waiting", "called before",
    # Money
    "my money", "money back", "pay for", "paying for", "paid",
    # Rhetorical anger
    "why", "how come", "how is this", "what kind of",
]

AGENT_EMPATHY_INDICATORS = [
    "i'm sorry", "i'm so sorry", "i apologize", "we apologize",
    "sorry about", "sorry for", "sorry to hear",
    "understand your", "i understand", "completely understand",
    "must be frustrating", "i can see how", "that must be",
    "appreciate your patience", "bear with",
]


def correct_sadness_for_cs(emotion: str, confidence: float, text: str,
                           speaker_role: str) -> tuple:
    """
    Context-aware sadness correction for customer-service transcripts.

    Returns (corrected_emotion, corrected_confidence, was_corrected, reason).
    """
    if emotion != "sad":
        return emotion, confidence, False, None

    text_lower = text.lower()

    # Agent saying empathetic things → not sad, it's professional empathy
    if speaker_role == "AGENT":
        for phrase in AGENT_EMPATHY_INDICATORS:
            if phrase in text_lower:
                return "neutral", max(confidence, 0.80), True, "agent_empathy→neutral"

    # Customer with frustration indicators → angry, not sad
    frustration_hits = sum(1 for p in FRUSTRATION_INDICATORS if p in text_lower)
    if frustration_hits >= 2:
        return "angry", confidence * 0.95, True, f"frustration({frustration_hits} hits)→angry"
    elif frustration_hits == 1:
        # Single hit: only remap if confidence was moderate (model was unsure)
        if confidence < 0.75:
            return "angry", confidence * 0.90, True, "mild_frustration→angry"

    # "sad" about apology text from agent without empathy phrases
    if speaker_role == "AGENT":
        return "neutral", confidence * 0.85, True, "agent_default→neutral"

    return emotion, confidence, False, None


# ==============================================================================
# FUSION CONFIGURATION
# ==============================================================================
FUSION_CONFIG = {
    'text_confident_threshold':  0.70,
    'audio_confident_threshold': 0.75,
    'audio_override_threshold':  0.85,

    'text_weight':  0.65,
    'audio_weight': 0.35,

    'short_segment_threshold':   0.8,
    'short_segment_text_weight': 0.85,
    'short_segment_audio_weight':0.15,

    'agent_audio_penalty': 0.4,
}


# ==============================================================================
# AUDIO EMOTION — wav2vec2 VAD regression (shared by V2 and V4)
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


class AudioEmotionClassifier:
    """Wav2Vec2 VAD regression → discrete label. Used by BOTH V2 and V4."""

    def __init__(self):
        print(f"  Loading Audio Emotion Model: {V2_AUDIO_EMOTION_MODEL}...")
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(V2_AUDIO_EMOTION_MODEL)
        self.model     = EmotionModel.from_pretrained(V2_AUDIO_EMOTION_MODEL)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        print("  [OK] Audio Emotion Model Loaded")

    @staticmethod
    def _dimensions_to_emotion(arousal: float, valence: float, dominance: float) -> tuple:
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
# V2 TEXT EMOTION — GoEmotions (28 classes)
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
# V4 TEXT EMOTION — j-hartmann (7 Ekman) + sadness correction
# ==============================================================================

class V4TextEmotionClassifier:
    """
    j-hartmann/emotion-english-distilroberta-base with CS sadness correction.

    Pipeline:
        1. Model predicts 7-class Ekman emotion
        2. Labels are normalized to unified set
        3. Sadness correction applied based on speaker role + context
    """

    def __init__(self):
        print(f"  Loading V4 Text Emotion Model: {V4_TEXT_EMOTION_MODEL}...")
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model=V4_TEXT_EMOTION_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None
        )
        print("  [OK] V4 Text Emotion Model Loaded")

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)',         r'$\1',   text)
        return " ".join(text.split())

    def predict(self, text: str, speaker_role: str = "UNKNOWN") -> Dict:
        return self.predict_batch([text], [speaker_role])[0]

    def predict_batch(self, texts: List[str],
                      speaker_roles: List[str] = None) -> List[Dict]:
        """
        Batch prediction with per-segment sadness correction.
        speaker_roles: parallel list of roles for each text.
        """
        if speaker_roles is None:
            speaker_roles = ["UNKNOWN"] * len(texts)

        cleaned = [self._clean(t) if t.strip() else "" for t in texts]
        results = self.classifier(cleaned)

        out = []
        for i, res in enumerate(results):
            if not cleaned[i]:
                out.append({'emotion': 'neutral', 'confidence': 0.0,
                            'all_scores': {}, 'corrected': False,
                            'correction_reason': None})
                continue

            # Normalize labels
            normalized = []
            for r in res:
                mapped = TEXT_LABEL_MAP.get(r['label'], r['label'])
                normalized.append({'label': mapped, 'score': r['score']})

            top        = max(normalized, key=lambda x: x['score'])
            all_scores = {r['label']: r['score'] for r in normalized}

            raw_emotion = top['label']
            raw_conf    = top['score']

            # Apply sadness correction
            corrected_emotion, corrected_conf, was_corrected, reason = \
                correct_sadness_for_cs(raw_emotion, raw_conf, texts[i],
                                       speaker_roles[i])

            out.append({
                'emotion':           corrected_emotion,
                'confidence':        corrected_conf,
                'all_scores':        all_scores,
                'raw_emotion':       raw_emotion if was_corrected else None,
                'corrected':         was_corrected,
                'correction_reason': reason,
            })
        return out


# ==============================================================================
# EMOTION FUSION
# ==============================================================================

def apply_fusion_logic(text_res: Dict, audio_res: Dict, text_content: str,
                       segment_duration: float = 1.0,
                       speaker_role: str = 'UNKNOWN') -> Dict:
    cfg = FUSION_CONFIG
    text_lower = text_content.lower()

    t_emotion, t_conf = text_res['emotion'], text_res['confidence']
    a_emotion, a_conf = audio_res['emotion'], audio_res['confidence']

    original_emotion = t_emotion
    was_corrected    = False

    # Rule-based text corrections (empathy / politeness misclassified)
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

    # Weight calculation
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

    # Fusion decision
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
        }
    }


# ==============================================================================
# SPEAKER ROLE CLASSIFIER  (identical to v2)
# ==============================================================================

class ProductionSpeakerRoleClassifier:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("   Initializing Production Speaker Role Classifier...")
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        print("   [1/4] Loading zero-shot LLM (BART-large-MNLI)...")
        zs_name      = "facebook/bart-large-mnli"
        zs_tokenizer = AutoTokenizer.from_pretrained(zs_name)
        zs_model     = AutoModelForSequenceClassification.from_pretrained(zs_name, use_safetensors=True)
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification", model=zs_model, tokenizer=zs_tokenizer,
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
        labels = ["a customer support representative", "a customer seeking help"]
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
                agent_score += 0.40; confidence = 0.95
            elif complaint_start:
                agent_score -= 0.30; confidence = 0.85
            else:
                agent_score += 0.20; confidence = 0.65

        first_three = " ".join(s.get('text', '') for s in speaker_segs[:3]).lower()
        if any(re.search(p, first_three) for p in self.greeting_patterns):
            agent_score += 0.15; confidence = max(confidence, 0.80)

        if len(speaker_segs) > 1:
            last_two = " ".join(s.get('text', '') for s in speaker_segs[-2:]).lower()
            if any(cp in last_two for cp in self.closing_patterns):
                agent_score += 0.15; confidence = max(confidence, 0.85)

        return {'agent_score': max(0.0, min(1.0, agent_score)), 'confidence': confidence}

    def analyze_linguistic_patterns(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        strong_a   = sum(1 for p in self.agent_indicators['strong']   if p in text_lower)
        moderate_a = sum(1 for p in self.agent_indicators['moderate'] if p in text_lower)
        agent_raw  = strong_a * 0.40 + moderate_a * 0.18
        strong_c   = sum(1 for p in self.customer_indicators['strong']   if p in text_lower)
        moderate_c = sum(1 for p in self.customer_indicators['moderate'] if p in text_lower)
        cust_raw   = strong_c * 0.40 + moderate_c * 0.18
        agent_score = max(0.0, min(1.0, 0.5 + agent_raw - cust_raw))
        if strong_a > 0 or strong_c > 0:     conf = 0.92
        elif moderate_a > 0 or moderate_c > 0: conf = 0.72
        else:                                  conf = 0.30
        return {'agent_score': agent_score, 'confidence': conf}

    def analyze_turn_taking(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        other_segs   = [s for s in segments if s.get('speaker') != speaker_id and s.get('speaker')]
        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        my_words    = np.mean([len(s.get('text', '').split()) for s in speaker_segs])
        other_words = np.mean([len(s.get('text', '').split()) for s in other_segs]) if other_segs else my_words
        ratio = my_words / other_words if other_words > 0 else 1.0

        if ratio > 1.3:    word_score, word_conf = 0.70, 0.65
        elif ratio < 0.75: word_score, word_conf = 0.30, 0.65
        else:              word_score, word_conf = 0.50, 0.35

        my_text    = " ".join(s.get('text', '') for s in speaker_segs)
        other_text = " ".join(s.get('text', '') for s in other_segs) if other_segs else ""
        my_q    = my_text.count('?')   / max(len(speaker_segs), 1)
        other_q = other_text.count('?')/ max(len(other_segs), 1) if other_segs else 0.0

        if my_q > other_q + 0.15:    q_score, q_conf = 0.65, 0.60
        elif my_q < other_q - 0.15:  q_score, q_conf = 0.35, 0.60
        else:                         q_score, q_conf = 0.50, 0.30

        my_solutions = sum(1 for p in self.solution_phrases if p in my_text.lower())
        if my_solutions >= 2:   sol_score, sol_conf = 0.85, 0.80
        elif my_solutions == 1: sol_score, sol_conf = 0.70, 0.55
        else:                   sol_score, sol_conf = 0.50, 0.25

        agent_score = (word_score * 0.30 + q_score * 0.25 + sol_score * 0.45)
        confidence  = (word_conf  * 0.30 + q_conf  * 0.25 + sol_conf  * 0.45)
        return {'agent_score': max(0.0, min(1.0, agent_score)), 'confidence': confidence}

    def ensemble_classification(self, zero_shot, structure, linguistic, turn_taking):
        components = {
            'zero_shot': zero_shot, 'conversation_structure': structure,
            'linguistic_patterns': linguistic, 'turn_taking': turn_taking,
        }
        weighted_sum = total_weight = 0.0
        debug_info = {}
        for name, comp in components.items():
            base_w = self.ensemble_weights[name]
            eff_w  = base_w * (0.5 + comp['confidence'] * 0.5)
            weighted_sum += comp['agent_score'] * eff_w
            total_weight += eff_w
            debug_info[name] = {'score': comp['agent_score'], 'confidence': comp['confidence'],
                                'effective_weight': eff_w}

        agent_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        if agent_prob > 0.55:    role, final_conf = "AGENT", agent_prob
        elif agent_prob < 0.45:  role, final_conf = "CUSTOMER", 1 - agent_prob
        elif structure['agent_score'] > 0.55: role, final_conf = "AGENT", 0.68
        else:                    role, final_conf = "CUSTOMER", 0.68

        debug_info['final'] = {'agent_probability': agent_prob, 'role': role,
                               'final_confidence': final_conf}
        return role, final_conf, debug_info

    def classify_speaker_role(self, speaker_id: str, segments: List[Dict]):
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        if not speaker_segs:
            return "UNKNOWN", 0.0, {}
        full_text = " ".join(s.get('text', '') for s in speaker_segs)
        return self.ensemble_classification(
            self.classify_with_zero_shot(full_text),
            self.analyze_conversation_structure(speaker_id, segments),
            self.analyze_linguistic_patterns(full_text),
            self.analyze_turn_taking(speaker_id, segments),
        )

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
            if results[speakers_list[0]] == results[speakers_list[1]]:
                all_spk = [s.get('speaker') for s in segments if s.get('speaker')]
                first_speaker = all_spk[0] if all_spk else speakers_list[0]
                print(f"\n   >> Both classified as {results[first_speaker]}. "
                      f"First-speaker heuristic: {first_speaker} -> AGENT")
                results[first_speaker] = 'AGENT'
                other = [s for s in speakers_list if s != first_speaker][0]
                results[other] = 'CUSTOMER'
        return results


# ==============================================================================
# OVERLAP DETECTION
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
# COMPARISON PIPELINE — V2 vs V4 (hybrid) side-by-side
# ==============================================================================

class ComparisonPipeline:
    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing V2 vs V4 Comparison Pipeline ===")

    def _load_whisper_and_diarize(self):
        print(">> Loading ASR & Diarization Models...")
        import whisperx
        from whisperx.diarize import DiarizationPipeline
        asr_model     = whisperx.load_model(self.whisper_model_size, DEVICE, compute_type=COMPUTE_TYPE)
        print("     [OK] WhisperX Loaded")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("     [OK] Diarization Loaded")
        return whisperx, asr_model, diarize_model

    def process(self, audio_path: str, language: str = None) -> Dict:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return {}

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

        # PHASE 2 — ROLE CLASSIFICATION
        print(">> Step 5: Speaker Role Classification...")
        role_clf = ProductionSpeakerRoleClassifier(device=DEVICE)
        flat_segments = [
            {"text": seg["text"].strip(), "speaker": seg.get("speaker", "UNKNOWN")}
            for seg in result["segments"]
        ]
        speaker_roles = role_clf.classify_all_speakers(flat_segments)
        del role_clf
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # PHASE 3 — EMOTION MODELS
        print("\n>> Step 6: Loading Emotion Models...")
        v2_text_clf = V2TextEmotionClassifier()
        v4_text_clf = V4TextEmotionClassifier()
        audio_clf   = AudioEmotionClassifier()  # shared by both

        print("\n>> Step 7: Running V2 vs V4 Comparison...")
        print("=" * 85)

        all_texts     = [seg["text"].strip() for seg in result["segments"]]
        all_roles     = [speaker_roles.get(seg.get("speaker", "UNKNOWN"), "UNKNOWN")
                         for seg in result["segments"]]

        v2_text_preds = v2_text_clf.predict_batch(all_texts)
        v4_text_preds = v4_text_clf.predict_batch(all_texts, all_roles)

        full_audio, _ = librosa.load(audio_path, sr=16000)

        segments_out = []
        stats = {'text_agree': 0, 'fusion_agree': 0, 'total': 0,
                 'v4_corrections': 0}

        for idx, seg in enumerate(result["segments"]):
            start    = seg["start"]
            end      = seg["end"]
            text     = seg["text"].strip()
            speaker  = seg.get("speaker", "UNKNOWN")
            role     = speaker_roles.get(speaker, "UNKNOWN")
            duration = end - start

            # Shared audio prediction
            audio_seg = full_audio[int(start * 16000):int(end * 16000)]
            audio_res = audio_clf.predict(audio_seg)

            # V2 fusion
            v2_text  = v2_text_preds[idx]
            v2_fusion = apply_fusion_logic(v2_text, audio_res, text,
                                           segment_duration=duration,
                                           speaker_role=role)

            # V4 fusion
            v4_text  = v4_text_preds[idx]
            v4_fusion = apply_fusion_logic(v4_text, audio_res, text,
                                           segment_duration=duration,
                                           speaker_role=role)

            # Stats
            stats['total'] += 1
            if v2_text['emotion'] == v4_text['emotion']:
                stats['text_agree'] += 1
            if v2_fusion['emotion'] == v4_fusion['emotion']:
                stats['fusion_agree'] += 1
            if v4_text.get('corrected'):
                stats['v4_corrections'] += 1

            seg_out = {
                "start": start, "end": end, "speaker": speaker,
                "role": role, "text": text, "overlap": seg.get('overlap', False),
                "audio": {
                    "emotion": audio_res['emotion'],
                    "confidence": audio_res['confidence'],
                    "dimensions": audio_res.get('dimensions', {}),
                },
                "v2": {
                    "text_emotion":  v2_text['emotion'],
                    "text_conf":     v2_text['confidence'],
                    "fused_emotion": v2_fusion['emotion'],
                    "fused_conf":    v2_fusion['confidence'],
                    "fused_source":  v2_fusion['source'],
                },
                "v4": {
                    "text_emotion":      v4_text['emotion'],
                    "text_conf":         v4_text['confidence'],
                    "text_raw":          v4_text.get('raw_emotion'),
                    "text_corrected":    v4_text.get('corrected', False),
                    "correction_reason": v4_text.get('correction_reason'),
                    "fused_emotion":     v4_fusion['emotion'],
                    "fused_conf":        v4_fusion['confidence'],
                    "fused_source":      v4_fusion['source'],
                },
            }
            segments_out.append(seg_out)
            self._print_comparison(seg_out)

        # SUMMARY
        elapsed = time.time() - start_time
        total   = stats['total']

        print("\n" + "=" * 85)
        print("COMPARISON SUMMARY:  V2 (GoEmotions + VAD)  vs  V4 (j-hartmann + VAD + sadness fix)")
        print("=" * 85)
        print(f"Total segments:      {total}")
        print(f"Text agree:          {stats['text_agree']}/{total} "
              f"({stats['text_agree']/total*100:.1f}%)")
        print(f"Fused agree:         {stats['fusion_agree']}/{total} "
              f"({stats['fusion_agree']/total*100:.1f}%)")
        print(f"V4 sadness fixes:    {stats['v4_corrections']}")
        print(f"Processing time:     {elapsed:.1f}s")

        # Confidence comparison
        v2_f = [s['v2']['fused_conf'] for s in segments_out]
        v4_f = [s['v4']['fused_conf'] for s in segments_out]
        print(f"\nAvg Fused Conf:      V2={np.mean(v2_f):.3f}  V4={np.mean(v4_f):.3f}")

        # Disagreement details
        disagree = [s for s in segments_out if s['v2']['fused_emotion'] != s['v4']['fused_emotion']]
        if disagree:
            print(f"\n--- Disagreements ({len(disagree)} segments) ---")
            for d in disagree:
                v4_fix = ""
                if d['v4']['text_corrected']:
                    v4_fix = f" [FIXED: {d['v4']['text_raw']}→{d['v4']['text_emotion']} ({d['v4']['correction_reason']})]"
                print(f"  [{d['start']:.1f}s-{d['end']:.1f}s] {d['role']}: "
                      f"\"{d['text'][:55]}...\"")
                print(f"    V2 fused: {d['v2']['fused_emotion']:>12s} ({d['v2']['fused_conf']:.1%})"
                      f"  |  V4 fused: {d['v4']['fused_emotion']:>10s} ({d['v4']['fused_conf']:.1%}){v4_fix}")

        return {'segments': segments_out, 'stats': stats, 'elapsed': elapsed}

    @staticmethod
    def _print_comparison(seg: Dict):
        v2, v4 = seg['v2'], seg['v4']
        text_match  = "==" if v2['text_emotion']  == v4['text_emotion']  else "!="
        fused_match = "==" if v2['fused_emotion'] == v4['fused_emotion'] else "!="
        marker = "  <<< DISAGREE" if fused_match == "!=" else ""

        fix_tag = ""
        if v4.get('text_corrected'):
            fix_tag = f" [was:{v4['text_raw']}→{v4['text_emotion']}]"

        print(f"\n[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['role']}"
              f"{'  [OVERLAP]' if seg.get('overlap') else ''}{marker}")
        print(f"   \"{seg['text'][:72]}{'...' if len(seg['text']) > 72 else ''}\"")
        print(f"   Audio:  {seg['audio']['emotion']:>10s} ({seg['audio']['confidence']:5.1%})")
        print(f"   Text:   V2={v2['text_emotion']:>12s} ({v2['text_conf']:5.1%})"
              f"  {text_match}  V4={v4['text_emotion']:>10s} ({v4['text_conf']:5.1%}){fix_tag}")
        print(f"   Fused:  V2={v2['fused_emotion']:>12s} ({v2['fused_conf']:5.1%})"
              f"  {fused_match}  V4={v4['fused_emotion']:>10s} ({v4['fused_conf']:5.1%})")


# ==============================================================================
# V4 STANDALONE PIPELINE  (drop-in replacement for main_v2)
# ==============================================================================

class V4Pipeline:
    """
    Production pipeline: j-hartmann text (with CS sadness correction) + audeering audio.
    Drop-in replacement for CombinedPipeline in main_v2.py.
    """

    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing V4 Hybrid Production Pipeline ===")

    def _load_whisper_and_diarize(self):
        print(">> Loading ASR & Diarization Models...")
        import whisperx
        from whisperx.diarize import DiarizationPipeline
        asr_model     = whisperx.load_model(self.whisper_model_size, DEVICE, compute_type=COMPUTE_TYPE)
        print("     [OK] WhisperX Loaded")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("     [OK] Diarization Loaded")
        return whisperx, asr_model, diarize_model

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

        # PHASE 2 — ROLE CLASSIFICATION
        print(">> Step 5: Speaker Role Classification...")
        role_clf = ProductionSpeakerRoleClassifier(device=DEVICE)
        flat_segments = [
            {"text": seg["text"].strip(), "speaker": seg.get("speaker", "UNKNOWN")}
            for seg in result["segments"]
        ]
        speaker_roles = role_clf.classify_all_speakers(flat_segments)
        del role_clf
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # PHASE 3 — EMOTION
        print("\n>> Step 6: Analyzing Emotions (V4 Hybrid)...")
        text_clf  = V4TextEmotionClassifier()
        audio_clf = AudioEmotionClassifier()

        all_texts = [seg["text"].strip() for seg in result["segments"]]
        all_roles = [speaker_roles.get(seg.get("speaker", "UNKNOWN"), "UNKNOWN")
                     for seg in result["segments"]]
        text_predictions = text_clf.predict_batch(all_texts, all_roles)

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
                "start": start, "end": end,
                "speaker": speaker, "role": role, "text": text,
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

        fix_tag = ""
        if text_res.get('corrected'):
            fix_tag = f" [was:{text_res['raw_emotion']}→{t_emo}]"
        corrected_str = (f" (was: {fusion['original_emotion']})"
                         if fusion.get('corrected') and fusion.get('original_emotion') else "")

        print(f"\n[{start:.1f}s-{end:.1f}s] {role}{overlap_tag}")
        print(f"   Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"   Text Emotion:  {t_emo:12s} ({t_conf:5.1%}){fix_tag}{corrected_str}")
        dims = audio_res.get('dimensions', {})
        dims_str = (f" [V:{dims.get('valence',0):.2f} "
                    f"A:{dims.get('arousal',0):.2f} "
                    f"D:{dims.get('dominance',0):.2f}]") if dims else ""
        print(f"   Audio Emotion: {a_emo:12s} ({a_conf:5.1%}){dims_str}")
        print(f"   >> Fused:      {f_emo:12s} ({f_conf:5.1%}) [{f_src}]")


# ==============================================================================
# JSON SERIALIZATION
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
        description="VocalMind V4 Hybrid Pipeline (j-hartmann text + audeering audio + sadness fix)")
    parser.add_argument("--audio",    type=str, default=DEFAULT_AUDIO_FILE,
                        help="Input audio file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output",   type=str,
                        help="Save results to JSON")
    parser.add_argument("--mode",     type=str, default="compare",
                        choices=["compare", "v4only"],
                        help="'compare' = V2 vs V4 side-by-side, "
                             "'v4only' = V4 hybrid pipeline only")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        sys.exit(1)

    if args.mode == "compare":
        pipeline = ComparisonPipeline()
        results  = pipeline.process(args.audio, language=args.language)
    else:
        pipeline = V4Pipeline()
        results  = pipeline.process(args.audio, language=args.language)

    if args.output and results:
        save_results_json(results, args.output)

    print("\n" + "=" * 50)
    print("DONE.")
