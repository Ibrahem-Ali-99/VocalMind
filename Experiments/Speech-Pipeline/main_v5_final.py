"""
VocalMind Pipeline v5 — Production-Ready Emotion Analysis (Final)
==================================================================

Final production pipeline combining best models + context-aware corrections:

Models:
    Text:  j-hartmann/emotion-english-distilroberta-base  (7 Ekman classes)
    Audio: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  (VAD regression)

Emotion correction layers:
    1. Customer frustration → anger remap (prevents sad misclassification)
    2. Gratitude detection (catches thank you, appreciate → happy)
    3. Agent empathy neutralization (scripted phrases like "I'm sorry" → neutral)
    4. Agent emotion PRESERVED for QA (detect if agent lost composure)

Key design decision:
    We DO NOT neutralize all agent emotions. If an agent shows anger/frustration,
    that's a quality signal we want to capture. We only neutralize empathy phrases
    because those are scripted professional responses, not actual agent emotion.

Usage:
    python main_v5_final.py --audio your_file.mp3 --output results.json
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
    print("⚠ WARNING: HF_TOKEN not found in .env")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Device: {DEVICE} ({COMPUTE_TYPE})")

warnings.filterwarnings("ignore")
if 'transformers' in sys.modules:
    transformers.logging.set_verbosity_error()

# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================

TEXT_EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"
AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

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
# Phrase sets for emotion correction
# ---------------------------------------------------------------------------

# Customer frustration indicators (sad → angry remap)
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

# Agent empathy indicators (scripted phrases → neutral)
AGENT_EMPATHY_INDICATORS = [
    "i'm sorry", "i'm so sorry", "i apologize", "we apologize",
    "sorry about", "sorry for", "sorry to hear",
    "understand your", "i understand", "completely understand",
    "must be frustrating", "i can see how", "that must be",
    "appreciate your patience", "bear with",
]

# Gratitude indicators (detect thank you expressions → happy)
GRATITUDE_INDICATORS = [
    "thank you", "thanks", "thank u", "thx", "tysm",
    "appreciate", "grateful", "gratitude",
    "thanks so much", "thank you so much", "really appreciate",
    "very grateful", "much appreciated",
]

# Positive closing phrases (boost happy confidence)
POSITIVE_PHRASES = [
    "have a nice day", "great day", "wonderful day",
    "take care", "good bye", "goodbye",
    "welcome", "my pleasure", "happy to help",
    "sounds good", "perfect", "great", "wonderful",
    "excellent", "awesome", "fantastic",
]


def apply_emotion_corrections(emotion: str, confidence: float, text: str,
                              speaker_role: str) -> tuple:
    """
    Context-aware emotion correction for customer-service transcripts.

    Corrections applied:
        1. Customer frustration (sad → angry)
        2. Gratitude detection (neutral/joy → happy with boost)
        3. Agent empathy phrases (any emotion → neutral for scripted responses)
        4. Agent emotions OTHERWISE PRESERVED (to detect loss of composure)

    Returns (corrected_emotion, corrected_confidence, was_corrected, reason).
    """
    text_lower = text.lower()
    original_emotion = emotion
    original_conf = confidence

    # ================================================================
    # LAYER 1: GRATITUDE DETECTION (both agent & customer)
    # ================================================================
    # Catches "thank you", "appreciate" → boost to happy
    # This runs FIRST because gratitude should override everything else
    # ================================================================
    gratitude_hits = sum(1 for p in GRATITUDE_INDICATORS if p in text_lower)
    if gratitude_hits > 0:
        # If model already said joy/happy, boost confidence
        if emotion in ['happy', 'joy']:
            return emotion, min(0.95, confidence * 1.15), True, f"gratitude_boost({gratitude_hits})"
        # If model said neutral/sad, remap to happy
        elif emotion in ['neutral', 'sad']:
            return 'happy', 0.85, True, f"gratitude_detect({gratitude_hits})→happy"

    # ================================================================
    # LAYER 2: CUSTOMER FRUSTRATION → ANGER
    # ================================================================
    # j-hartmann over-predicts "sad" for frustrated customer demands
    # Only applies to CUSTOMER role
    # ================================================================
    if speaker_role == "CUSTOMER" and emotion == "sad":
        frustration_hits = sum(1 for p in FRUSTRATION_INDICATORS if p in text_lower)

        if frustration_hits >= 2:
            # Strong frustration → definitely angry
            return "angry", confidence * 0.95, True, f"frustration({frustration_hits} hits)→angry"
        elif frustration_hits == 1 and confidence < 0.75:
            # Mild frustration + low confidence → probably angry
            return "angry", confidence * 0.90, True, "mild_frustration→angry"

    # ================================================================
    # LAYER 3: AGENT EMPATHY PHRASES → NEUTRAL
    # ================================================================
    # Scripted professional responses like "I'm sorry to hear that"
    # are not actual agent emotion — they're training/scripts
    # This does NOT neutralize all agent emotions (see design doc above)
    # ================================================================
    if speaker_role == "AGENT":
        for phrase in AGENT_EMPATHY_INDICATORS:
            if phrase in text_lower:
                return "neutral", max(confidence, 0.85), True, "agent_empathy_script→neutral"

    # ================================================================
    # NO CORRECTION NEEDED
    # ================================================================
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
# AUDIO EMOTION — wav2vec2 VAD regression
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
    """Wav2Vec2 VAD regression → discrete label."""

    def __init__(self):
        print(f"Loading Audio Emotion Model: {AUDIO_EMOTION_MODEL}...")
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(AUDIO_EMOTION_MODEL)
        self.model     = EmotionModel.from_pretrained(AUDIO_EMOTION_MODEL)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        print("[OK] Audio Emotion Model Loaded")

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
# TEXT EMOTION — j-hartmann + correction layers
# ==============================================================================

class TextEmotionClassifier:
    """
    j-hartmann/emotion-english-distilroberta-base + CS correction layers.
    """

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

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)',         r'$\1',   text)
        return " ".join(text.split())

    def predict(self, text: str, speaker_role: str = "UNKNOWN") -> Dict:
        return self.predict_batch([text], [speaker_role])[0]

    def predict_batch(self, texts: List[str],
                      speaker_roles: List[str] = None) -> List[Dict]:
        """Batch prediction with per-segment corrections."""
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

            # Apply correction layers
            corrected_emotion, corrected_conf, was_corrected, reason = \
                apply_emotion_corrections(raw_emotion, raw_conf, texts[i],
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
    """Combine text and audio emotion predictions."""
    cfg = FUSION_CONFIG

    t_emotion, t_conf = text_res['emotion'], text_res['confidence']
    a_emotion, a_conf = audio_res['emotion'], audio_res['confidence']

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
        'emotion':      fused_emotion,
        'confidence':   fused_conf,
        'source':       source,
        'text_details': {
            'emotion':           text_res['emotion'],
            'confidence':        text_res['confidence'],
            'all_scores':        text_res.get('all_scores', {}),
            'raw_emotion':       text_res.get('raw_emotion'),
            'corrected':         text_res.get('corrected', False),
            'correction_reason': text_res.get('correction_reason'),
        },
        'audio_details': {
            'emotion':    audio_res['emotion'],
            'confidence': audio_res['confidence'],
            'dimensions': audio_res.get('dimensions', {}),
        }
    }


# ==============================================================================
# SPEAKER ROLE CLASSIFIER
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
            r"(welcome to)", r"(this is\s+\w+\s+from)",
            r"(i'll be (helping|assisting) you)",
        ]
        self.closing_patterns = [
            "anything else", "is there anything", "have a great day",
            "thank you for calling", "take care", "goodbye",
            "glad i could help",
        ]

        print("   [3/4] Initializing industry-grade linguistic patterns...")
        self.agent_indicators = {
            'strong': [
                "i can see your account", "looking at your account",
                "system shows", "how can i help", "thank you for calling",
                "you've reached", "my name is", "i'll be assisting",
                "let me transfer", "ticket number", "case number",
                "can i verify", "for security purposes",
                "here's what we can do", "the steps are",
                "let me walk you through", "we are aware",
            ],
            'moderate': [
                "let me check", "i understand your", "i apologize for",
                "let me help you", "happy to help", "i can see",
                "let me look into", "one moment please",
                "bear with me", "please hold",
            ],
        }
        self.customer_indicators = {
            'strong': [
                "my phone", "my account", "my bill", "my order",
                "i have a problem", "i need help", "not working",
                "this is ridiculous", "i want to cancel", "i want a refund",
                "i was charged", "i didn't receive",
            ],
            'moderate': [
                "can you help", "why is", "how do i",
                "i'm frustrated", "this keeps happening",
                "i called before", "still not fixed",
            ],
        }

        print("   [4/4] Configuring ensemble weights...")
        self.solution_phrases = [
            "try restarting", "clear the cache", "go to settings",
            "you need to", "you should", "the steps are",
            "uninstall", "reinstall", "update",
        ]
        self.ensemble_weights = {
            'zero_shot': 0.35, 'conversation_structure': 0.25,
            'linguistic_patterns': 0.23, 'turn_taking': 0.17,
        }
        print("   [OK] Production classifier ready\n")

    def classify_with_zero_shot(self, text: str) -> Dict[str, float]:
        labels = ["a customer support representative", "a customer seeking help"]
        try:
            result = self.zero_shot_classifier(text[:1000], labels, multi_label=False)
            agent_prob = result['scores'][0] if result['labels'][0] == labels[0] else result['scores'][1]
            return {'agent_score': agent_prob, 'confidence': max(result['scores'])}
        except Exception as e:
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
                "my phone", "my account", "not working", "i need help", "i have a problem"
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
        conf = 0.92 if (strong_a > 0 or strong_c > 0) else (0.72 if (moderate_a > 0 or moderate_c > 0) else 0.30)
        return {'agent_score': agent_score, 'confidence': conf}

    def analyze_turn_taking(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        other_segs   = [s for s in segments if s.get('speaker') != speaker_id and s.get('speaker')]
        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        my_words = np.mean([len(s.get('text', '').split()) for s in speaker_segs])
        other_words = np.mean([len(s.get('text', '').split()) for s in other_segs]) if other_segs else my_words
        ratio = my_words / other_words if other_words > 0 else 1.0

        if ratio > 1.3: word_score, word_conf = 0.70, 0.65
        elif ratio < 0.75: word_score, word_conf = 0.30, 0.65
        else: word_score, word_conf = 0.50, 0.35

        my_text = " ".join(s.get('text', '') for s in speaker_segs)
        my_solutions = sum(1 for p in self.solution_phrases if p in my_text.lower())
        if my_solutions >= 2: sol_score, sol_conf = 0.85, 0.80
        elif my_solutions == 1: sol_score, sol_conf = 0.70, 0.55
        else: sol_score, sol_conf = 0.50, 0.25

        agent_score = (word_score * 0.50 + sol_score * 0.50)
        confidence  = (word_conf  * 0.50 + sol_conf  * 0.50)
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
        if agent_prob > 0.55: role, final_conf = "AGENT", agent_prob
        elif agent_prob < 0.45: role, final_conf = "CUSTOMER", 1 - agent_prob
        elif structure['agent_score'] > 0.55: role, final_conf = "AGENT", 0.68
        else: role, final_conf = "CUSTOMER", 0.68

        debug_info['final'] = {'agent_probability': agent_prob, 'role': role, 'final_confidence': final_conf}
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
        print("   Classifying speaker roles...")
        print("   " + "=" * 60)
        for spk in speakers:
            role, conf, debug = self.classify_speaker_role(spk, segments)
            results[spk] = role
            print(f"\n   {spk}: {role} (confidence: {conf:.2f})")
        print("   " + "=" * 60)

        if len(results) == 2:
            speakers_list = list(results.keys())
            if results[speakers_list[0]] == results[speakers_list[1]]:
                all_spk = [s.get('speaker') for s in segments if s.get('speaker')]
                first_speaker = all_spk[0] if all_spk else speakers_list[0]
                print(f"\n   >> Both same role. First-speaker heuristic: {first_speaker} → AGENT")
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
# MAIN PIPELINE
# ==============================================================================

class ProductionPipeline:
    """
    Final production pipeline with all improvements:
        - j-hartmann text emotion (7 Ekman classes)
        - audeering audio emotion (VAD regression)
        - Customer frustration → anger correction
        - Gratitude detection
        - Agent empathy neutralization (NOT all agent emotions)
    """

    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing VocalMind Production Pipeline v5 ===")

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
            print(f"❌ File not found: {audio_path}")
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
        print("     [OK] Role classifier freed")

        # PHASE 3 — EMOTION ANALYSIS
        print("\n>> Step 6: Analyzing Emotions...")
        text_clf  = TextEmotionClassifier()
        audio_clf = AudioEmotionClassifier()

        all_texts = [seg["text"].strip() for seg in result["segments"]]
        all_roles = [speaker_roles.get(seg.get("speaker", "UNKNOWN"), "UNKNOWN")
                     for seg in result["segments"]]
        text_predictions = text_clf.predict_batch(all_texts, all_roles)

        full_audio, _ = librosa.load(audio_path, sr=16000)

        final_segments = []
        correction_stats = {'frustration': 0, 'gratitude': 0, 'empathy': 0}

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

            # Track correction stats
            if text_res.get('corrected'):
                reason = text_res.get('correction_reason', '')
                if 'frustration' in reason:
                    correction_stats['frustration'] += 1
                elif 'gratitude' in reason:
                    correction_stats['gratitude'] += 1
                elif 'empathy' in reason:
                    correction_stats['empathy'] += 1

            final_segments.append({
                "start": start, "end": end,
                "speaker": speaker, "role": role, "text": text,
                "emotion_analysis": fusion,
            })

            self._print_utterance(speaker, role, text, start, end,
                                  fusion, is_overlap=seg.get('overlap', False))

        elapsed = time.time() - start_time
        print(f"\n>> Total processing time: {elapsed:.1f}s")
        print(">> Emotion corrections applied:")
        print(f"     Frustration→anger:  {correction_stats['frustration']}")
        print(f"     Gratitude detected: {correction_stats['gratitude']}")
        print(f"     Agent empathy→neutral: {correction_stats['empathy']}")

        return final_segments

    @staticmethod
    def _print_utterance(speaker, role, text, start, end, fusion, is_overlap=False):
        overlap_tag = " [OVERLAP]" if is_overlap else ""

        text_det = fusion['text_details']
        audio_det = fusion['audio_details']

        t_emo = text_det['emotion']
        t_conf = text_det['confidence']
        a_emo = audio_det['emotion']
        a_conf = audio_det['confidence']
        f_emo = fusion['emotion']
        f_conf = fusion['confidence']
        f_src = fusion['source']

        # Show corrections
        fix_tag = ""
        if text_det.get('corrected'):
            raw = text_det.get('raw_emotion')
            reason = text_det.get('correction_reason', '')
            fix_tag = f" [FIXED: {raw}→{t_emo} ({reason})]"

        dims = audio_det.get('dimensions', {})
        dims_str = (f" [V:{dims.get('valence',0):.2f} "
                    f"A:{dims.get('arousal',0):.2f} "
                    f"D:{dims.get('dominance',0):.2f}]") if dims else ""

        print(f"\n[{start:.1f}s-{end:.1f}s] {role}{overlap_tag}")
        print(f"   Text: \"{text[:68]}{'...' if len(text) > 68 else ''}\"")
        print(f"   Text Emotion:  {t_emo:12s} ({t_conf:5.1%}){fix_tag}")
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
    print(f"\n✓ Results saved to: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VocalMind Production Pipeline v5 — Final Emotion Analysis System")
    parser.add_argument("--audio",    type=str, default=DEFAULT_AUDIO_FILE,
                        help="Input audio file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output",   type=str,
                        help="Save results to JSON")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)

    pipeline = ProductionPipeline()
    results  = pipeline.process(args.audio, language=args.language)

    if args.output and results:
        save_results_json(results, args.output)

    print("\n" + "=" * 50)
    print("✓ DONE.")
