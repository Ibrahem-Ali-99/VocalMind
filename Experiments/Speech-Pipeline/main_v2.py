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

# PATCH 1: torch.load weights_only fix
try:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    print("[OK] torch.load patch applied")
except Exception:
    pass

# PATCH 2: torchaudio 2.11+ missing attributes
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

# PATCH 3: huggingface_hub use_auth_token deprecation
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
# 2. EMOTION MODELS
# ==============================================================================

TEXT_EMOTION_MODEL  = "SamLowe/roberta-base-go_emotions"
AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# ---------------------------------------------------------------------------
# Empathy / positive phrase sets (used in fusion correction)
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

    # Agent audio is typically less emotionally expressive (especially TTS-
    # generated agent lines), so we down-weight audio for agent segments to
    # avoid spurious emotion labels from flat prosody.
    'agent_audio_penalty': 0.4,
}


# ==============================================================================
# AUDIO EMOTION — wav2vec2 regression head + VAD → label mapping
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
    """Wav2Vec2-based audio emotion classifier outputting VAD dimensions."""

    def __init__(self):
        print(f"Loading Audio Emotion Model: {AUDIO_EMOTION_MODEL}...")
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(AUDIO_EMOTION_MODEL)
        self.model     = EmotionModel.from_pretrained(AUDIO_EMOTION_MODEL)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        print("[OK] Audio Emotion Model Loaded")

    # ------------------------------------------------------------------
    # VAD → discrete emotion label
    # ------------------------------------------------------------------
    @staticmethod
    def _dimensions_to_emotion(arousal: float, valence: float, dominance: float) -> Tuple[str, float]:
        """
        Map continuous Arousal / Valence / Dominance into a discrete emotion
        label plus a scalar confidence.  Fixed: every branch now returns an
        explicit (label, confidence) tuple — the previous version had an
        ambiguous ternary that produced an unintended tuple due to operator
        precedence.
        """
        if arousal > 0.7:
            if valence > 0.5:
                return ("happy",   valence * 0.9 + arousal * 0.1)
            else:
                if dominance > 0.5:
                    return ("angry",   (1 - valence) * 0.8 + arousal * 0.2)
                else:
                    return ("fearful", (1 - valence) * 0.7 + arousal * 0.3)

        if valence > 0.6:
            # Fixed: was `return "happy" if … else "neutral", valence`
            # which parsed as  ("happy", valence)  OR  ("neutral", valence)
            # depending on arousal — the ternary only guarded the first element.
            # Use >= 0.5 for arousal: VAD convention treats 0.5 as the active
            # boundary, so high valence + mid arousal maps to happy.
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

    # ------------------------------------------------------------------
    def predict(self, audio_array: np.ndarray, sr: int = 16000) -> Dict:
        min_samples = sr  # 1 second minimum
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
# TEXT EMOTION — RoBERTa go_emotions  (supports BATCH inference)
# ==============================================================================

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

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
        text = re.sub(r'\$\s*(\d)',         r'$\1',   text)
        return " ".join(text.split())

    def predict(self, text: str) -> Dict:
        """Single-text prediction (kept for backward compat)."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Batch prediction — single forward pass through the pipeline.
        ~3-4× faster than calling predict() in a loop when len(texts) > 4.
        """
        cleaned = [self._clean(t) if t.strip() else "" for t in texts]
        # pipeline accepts a list and returns a list-of-lists
        results = self.classifier(cleaned)          # List[List[{label, score}]]

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
# EMOTION FUSION
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
    if t_emotion in {'anger', 'sadness', 'disgust', 'fear', 'annoyance', 'disapproval'}:
        for phrase in EMPATHY_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion     = 'neutral'
                t_conf        = max(t_conf, 0.85)
                was_corrected = True
                break

    if t_emotion in {'fear', 'anger', 'disgust', 'sadness', 'annoyance'}:
        for phrase in POSITIVE_PHRASE_PATTERNS:
            if phrase in text_lower:
                t_emotion     = 'gratitude' if 'thank' in phrase else 'neutral'
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
        # Agent audio is often flat / TTS-generated → down-weight
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

        # Both models uncertain → default neutral, cap confidence honestly
        if t_conf < 0.50 and a_conf_eff < 0.50:
            fused_emotion = 'neutral'
            fused_conf    = 0.40          # was max(t_conf, a_eff) — could be 0.49, misleading
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
            'dimensions': audio_res.get('dimensions', {})
        }
    }


# ==============================================================================
# PRODUCTION-GRADE SPEAKER ROLE CLASSIFIER  (4-component ensemble)
# ==============================================================================
# Removed in this version (vs. previous):
#   • sentiment_analyzer  (cardiffnlp RoBERTa) — 0.05 weight, ~200 MB
#   • question_classifier (shahrukhx01)        — 0.05 weight, ~70  MB
# Their combined weight (0.10) is redistributed:
#   zero_shot  0.30 → 0.35   (+0.05)
#   linguistic 0.20 → 0.23   (+0.03)
#   turn_taking 0.15 → 0.17  (+0.02)
# This also removes ~1-1.5 s of model-load time and per-call inference.
# ==============================================================================

class ProductionSpeakerRoleClassifier:
    """
    4-component ensemble for speaker-role classification:
        1. Zero-shot LLM  (BART-large-MNLI)   — primary semantic signal
        2. Conversation-structure analysis     — position / greeting / closing
        3. Linguistic-pattern matching         — industry-grade indicator sets
        4. Relative turn-taking dynamics       — word-count ratio + solution detection

    Ensemble uses confidence-weighted voting.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("   Initializing Production Speaker Role Classifier...")

        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        # ----------------------------------------------------------------
        # 1. Zero-shot classifier  (PRIMARY)
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # 2. Conversation-structure patterns (no model — pure heuristic)
        # ----------------------------------------------------------------
        print("   [2/4] Initializing conversation-structure analyzer...")

        # Greeting patterns — covers real-world openings broadly
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

        # Closing patterns — agents typically close the interaction
        self.closing_patterns = [
            "anything else", "is there anything", "have a great day",
            "have a wonderful day", "thank you for calling",
            "thank you for contacting", "take care", "goodbye",
            "i hope that helps", "is there anything else i can do",
            "glad i could help", "hope we got that sorted",
        ]

        # ----------------------------------------------------------------
        # 3. Industry-grade indicator sets  (AGENT / CUSTOMER)
        # ----------------------------------------------------------------
        print("   [3/4] Initializing industry-grade linguistic patterns...")

        #
        # AGENT indicators — drawn from real contact-centre transcripts
        # Strong  : almost exclusively used by agents
        # Moderate: common in agent speech, occasional in customer speech
        #
        self.agent_indicators = {
            'strong': [
                # Account / system access
                "i can see your account",  "looking at your account",
                "pulling up your account", "i'm seeing here",
                "on my end",               "system shows",
                "i have your account",     "account is showing",
                # Procedural / scripted
                "how can i help",          "how may i help",
                "thank you for calling",   "thank you for contacting",
                "you've reached",          "this is tech support",
                "my name is",              "i'll be assisting",
                # Escalation / resolution vocabulary
                "let me transfer",         "i'll escalate",
                "ticket number",           "case number",
                "reference number",        "i'll create a ticket",
                # Verification / security
                "can i verify",            "for security purposes",
                "can you confirm your",    "verify your identity",
                "date of birth please",    "account holder",
                # Solution delivery
                "here's what we can do",   "the steps are",
                "you should see",          "that should fix",
                "let me walk you through", "i'd like to help you",
                # Status reporting
                "there is currently",      "we are aware",
                "known issue",             "our team is working",
                "estimated time",          "scheduled maintenance",
            ],
            'moderate': [
                "let me check",            "i understand your",
                "i apologize for",         "let me help you",
                "i'll assist you",         "happy to help",
                "according to",            "shows that",
                "i can see",               "on our side",
                "let me look into",        "i'll look into that",
                "bear with me",            "one moment please",
                "i appreciate your",       "bear with me while",
                "just a moment",           "please hold",
                "i'll get that sorted",    "let me arrange",
            ],
        }

        #
        # CUSTOMER indicators
        # Strong  : almost exclusively used by customers
        # Moderate: common in customer speech
        #
        self.customer_indicators = {
            'strong': [
                # Ownership / possession (personal problem)
                "my phone",               "my account",
                "my bill",                "my order",
                "my internet",            "my service",
                "my subscription",        "my card",
                # Problem statements
                "i have a problem",       "i need help",
                "keeps crashing",         "not working",
                "i'm having trouble",     "it stopped working",
                "it's not working",       "it won't",
                "doesn't work",           "broke down",
                # Frustration / escalation demands
                "this is ridiculous",     "this is unacceptable",
                "i've been waiting",      "i want to speak to",
                "i want to cancel",       "cancel my",
                "i want a refund",        "give me a refund",
                "i was charged",          "charged me",
                "i didn't receive",       "never received",
                # Warranty / entitlement
                "under warranty",         "still under warranty",
                "i paid for",             "i'm paying for",
            ],
            'moderate': [
                "can you help",           "why is",
                "how do i",               "i've tried",
                "i'm frustrated",         "i'm upset",
                "nobody helps",           "this keeps happening",
                "again?",                 "i already told",
                "i called before",        "last time",
                "still not fixed",        "same problem",
                "what's going on",        "what happened to",
            ],
        }

        # ----------------------------------------------------------------
        # 4. Turn-taking — no model, computed at classify time
        # ----------------------------------------------------------------
        print("   [4/4] Configuring ensemble weights...")

        # Solution-delivery phrases (agents provide these; customers don't)
        self.solution_phrases = [
            "try restarting",       "clear the cache",
            "go to settings",       "click on",
            "you need to",          "you should",
            "the steps are",       "first,",
            "next,",               "then,",
            "make sure",           "uninstall",
            "reinstall",           "update",
            "download",            "enable",
            "disable",             "change your",
            "reset your",          "restart your",
            "check your",          "navigate to",
            "select",              "tap on",
            "press",               "enter your",
        ]

        # Ensemble weights (redistributed after removing sentiment + question models)
        self.ensemble_weights = {
            'zero_shot':             0.35,
            'conversation_structure':0.25,
            'linguistic_patterns':   0.23,
            'turn_taking':           0.17,
        }

        print("   [OK] Production classifier ready\n")

    # ==================================================================
    # COMPONENT 1 — Zero-shot (BART-large-MNLI)
    # ==================================================================
    def classify_with_zero_shot(self, text: str) -> Dict[str, float]:
        """
        Shorter, more discriminative labels perform better with BART NLI.
        """
        labels = [
            "a customer support representative",
            "a customer seeking help"
        ]
        try:
            result = self.zero_shot_classifier(text[:1000], labels, multi_label=False)
            agent_prob = result['scores'][0] if result['labels'][0] == labels[0] else result['scores'][1]
            return {'agent_score': agent_prob, 'confidence': max(result['scores'])}
        except Exception as e:
            print(f"      ⚠ Zero-shot failed: {e}")
            return {'agent_score': 0.5, 'confidence': 0.0}

    # ==================================================================
    # COMPONENT 2 — Conversation structure
    # ==================================================================
    def analyze_conversation_structure(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        """
        Position-based + pattern-based structural signals.
        Real-world nuance:
          • First speaker in a support call is usually the AGENT (greeting).
          • But if first utterance is a complaint → that speaker is the CUSTOMER
            (e.g. live-chat where customer initiates).
          • Closing patterns (is there anything else?) strongly signal AGENT.
        """
        all_speakers     = [s.get('speaker') for s in segments if s.get('speaker')]
        speaker_segs     = [s for s in segments if s.get('speaker') == speaker_id]

        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        agent_score = 0.5
        confidence  = 0.0

        # ── Feature A: first-speaker position ──────────────────────────
        if all_speakers and speaker_id == all_speakers[0]:
            first_text = speaker_segs[0].get('text', '').lower()

            has_greeting  = any(re.search(p, first_text) for p in self.greeting_patterns)
            # If first utterance looks like a complaint, NOT an agent opening
            complaint_start = any(kw in first_text for kw in [
                "my phone", "my account", "not working", "i need help",
                "i have a problem", "keeps crashing", "i'm having trouble",
                "i want to", "this is ridiculous", "i called"
            ])

            if has_greeting and not complaint_start:
                agent_score += 0.40
                confidence   = 0.95
            elif complaint_start:
                agent_score -= 0.30          # this speaker is likely the customer
                confidence   = 0.85
            else:
                agent_score += 0.20          # mild first-speaker bonus
                confidence   = 0.65

        # ── Feature B: greeting in first 3 utterances ──────────────────
        first_three = " ".join(s.get('text', '') for s in speaker_segs[:3]).lower()
        if any(re.search(p, first_three) for p in self.greeting_patterns):
            agent_score += 0.15
            confidence   = max(confidence, 0.80)

        # ── Feature C: closing patterns (strong agent signal) ──────────
        if len(speaker_segs) > 1:
            last_two = " ".join(s.get('text', '') for s in speaker_segs[-2:]).lower()
            if any(cp in last_two for cp in self.closing_patterns):
                agent_score += 0.15
                confidence   = max(confidence, 0.85)

        agent_score = max(0.0, min(1.0, agent_score))
        return {'agent_score': agent_score, 'confidence': confidence}

    # ==================================================================
    # COMPONENT 3 — Linguistic pattern matching (industry-grade)
    # ==================================================================
    def analyze_linguistic_patterns(self, text: str) -> Dict[str, float]:
        """
        Weighted scoring across strong / moderate tiers.
        Weak-tier removed — they added noise with minimal signal.
        """
        text_lower = text.lower()

        # --- AGENT ---
        strong_a   = sum(1 for p in self.agent_indicators['strong']   if p in text_lower)
        moderate_a = sum(1 for p in self.agent_indicators['moderate'] if p in text_lower)
        agent_raw  = strong_a * 0.40 + moderate_a * 0.18

        # --- CUSTOMER ---
        strong_c   = sum(1 for p in self.customer_indicators['strong']   if p in text_lower)
        moderate_c = sum(1 for p in self.customer_indicators['moderate'] if p in text_lower)
        cust_raw   = strong_c * 0.40 + moderate_c * 0.18

        # Net score: 0.5 = neutral
        agent_score = max(0.0, min(1.0, 0.5 + agent_raw - cust_raw))

        # Confidence calibration based on tier hit
        if strong_a > 0 or strong_c > 0:
            conf = 0.92
        elif moderate_a > 0 or moderate_c > 0:
            conf = 0.72
        else:
            conf = 0.30          # no hits → low confidence

        return {'agent_score': agent_score, 'confidence': conf}

    # ==================================================================
    # COMPONENT 4 — Relative turn-taking + solution detection
    # ==================================================================
    def analyze_turn_taking(self, speaker_id: str, segments: List[Dict]) -> Dict[str, float]:
        """
        Compares THIS speaker's stats against the OTHER speaker(s) rather than
        using absolute thresholds.  Also counts solution-delivery phrases —
        only agents deliver step-by-step solutions.

        Signals:
          • Relative avg-words-per-turn  (agents tend to be longer)
          • Relative question density    (agents ask diagnostic questions)
          • Solution-phrase count        (agents deliver fixes)
        """
        all_speakers   = list({s.get('speaker') for s in segments if s.get('speaker')})
        speaker_segs   = [s for s in segments if s.get('speaker') == speaker_id]
        other_segs     = [s for s in segments if s.get('speaker') != speaker_id and s.get('speaker')]

        if not speaker_segs:
            return {'agent_score': 0.5, 'confidence': 0.0}

        # ── avg words per turn (relative) ──────────────────────────────
        my_words   = np.mean([len(s.get('text', '').split()) for s in speaker_segs])
        other_words = np.mean([len(s.get('text', '').split()) for s in other_segs]) if other_segs else my_words

        if other_words > 0:
            ratio = my_words / other_words       # >1 → this speaker talks more per turn
        else:
            ratio = 1.0

        if ratio > 1.3:
            word_score, word_conf = 0.70, 0.65
        elif ratio < 0.75:
            word_score, word_conf = 0.30, 0.65
        else:
            word_score, word_conf = 0.50, 0.35

        # ── question density (relative) ────────────────────────────────
        my_text   = " ".join(s.get('text', '') for s in speaker_segs)
        other_text= " ".join(s.get('text', '') for s in other_segs) if other_segs else ""

        my_q   = my_text.count('?')   / max(len(speaker_segs), 1)
        other_q= other_text.count('?')/ max(len(other_segs), 1) if other_segs else 0.0

        if my_q > other_q + 0.15:        # I ask notably more questions
            q_score, q_conf = 0.65, 0.60
        elif my_q < other_q - 0.15:
            q_score, q_conf = 0.35, 0.60
        else:
            q_score, q_conf = 0.50, 0.30

        # ── solution-phrase count ──────────────────────────────────────
        my_solutions = sum(1 for p in self.solution_phrases if p in my_text.lower())
        if my_solutions >= 2:
            sol_score, sol_conf = 0.85, 0.80
        elif my_solutions == 1:
            sol_score, sol_conf = 0.70, 0.55
        else:
            sol_score, sol_conf = 0.50, 0.25

        # ── weighted sub-combination ───────────────────────────────────
        # Solution detection is the strongest single signal here
        agent_score = (word_score * 0.30 + q_score * 0.25 + sol_score * 0.45)
        confidence  = (word_conf  * 0.30 + q_conf  * 0.25 + sol_conf  * 0.45)

        agent_score = max(0.0, min(1.0, agent_score))
        return {'agent_score': agent_score, 'confidence': confidence}

    # ==================================================================
    # ENSEMBLE FUSION
    # ==================================================================
    def ensemble_classification(self,
                                zero_shot:  Dict,
                                structure:  Dict,
                                linguistic: Dict,
                                turn_taking:Dict) -> Tuple[str, float, Dict]:
        """
        Confidence-weighted voting across 4 components.
        """
        components = {
            'zero_shot':              zero_shot,
            'conversation_structure': structure,
            'linguistic_patterns':    linguistic,
            'turn_taking':            turn_taking,
        }

        weighted_sum  = 0.0
        total_weight  = 0.0
        debug_info    = {}

        for name, comp in components.items():
            base_w = self.ensemble_weights[name]
            score  = comp['agent_score']
            conf   = comp['confidence']

            # effective weight ramps from base_w*0.5 (conf=0) to base_w*1.0 (conf=1)
            eff_w = base_w * (0.5 + conf * 0.5)

            weighted_sum += score * eff_w
            total_weight += eff_w

            debug_info[name] = {
                'score':            score,
                'confidence':       conf,
                'effective_weight': eff_w
            }

        agent_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

        # ── decision with hysteresis band ──────────────────────────────
        if agent_prob > 0.55:
            role, final_conf = "AGENT", agent_prob
        elif agent_prob < 0.45:
            role, final_conf = "CUSTOMER", 1 - agent_prob
        else:
            # Tie-break: lean on conversation structure (most reliable positional signal)
            if structure['agent_score'] > 0.55:
                role, final_conf = "AGENT",    0.68
            else:
                role, final_conf = "CUSTOMER", 0.68

        debug_info['final'] = {
            'agent_probability': agent_prob,
            'role':              role,
            'final_confidence':  final_conf,
        }
        return role, final_conf, debug_info

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def classify_speaker_role(self, speaker_id: str, segments: List[Dict]) -> Tuple[str, float, Dict]:
        speaker_segs = [s for s in segments if s.get('speaker') == speaker_id]
        if not speaker_segs:
            return "UNKNOWN", 0.0, {}

        full_text = " ".join(s.get('text', '') for s in speaker_segs)

        zs  = self.classify_with_zero_shot(full_text)
        st  = self.analyze_conversation_structure(speaker_id, segments)
        ling= self.analyze_linguistic_patterns(full_text)
        tt  = self.analyze_turn_taking(speaker_id, segments)

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
                    print(f"      • {comp_name:25s}: score={info['score']:.2f}, "
                          f"conf={info['confidence']:.2f}, weight={info['effective_weight']:.3f}")

        print("   " + "=" * 60)

        # ── post-processing: ensure role diversity ────────────────────
        if len(results) == 2:
            speakers_list = list(results.keys())
            roles         = [results[s] for s in speakers_list]

            if roles[0] == roles[1]:
                # Both same role → use first-speaker heuristic as fallback
                all_spk      = [s.get('speaker') for s in segments if s.get('speaker')]
                first_speaker= all_spk[0] if all_spk else speakers_list[0]

                print(f"\n   >> Both classified as {roles[0]}. "
                      f"Applying first-speaker heuristic: {first_speaker} → AGENT")

                results[first_speaker] = 'AGENT'
                other = [s for s in speakers_list if s != first_speaker][0]
                results[other]         = 'CUSTOMER'

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
            nxt ['overlap'] = True
    return segments


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class CombinedPipeline:
    def __init__(self, whisper_model_size: str = "large-v2"):
        self.whisper_model_size = whisper_model_size
        print("\n=== Initializing Production Pipeline ===")

    # ------------------------------------------------------------------
    # Phase 1 helpers
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
    # Phase 2 helpers
    # ------------------------------------------------------------------
    def _load_analysis_models(self):
        print(">> Loading Analysis Models (Emotion & Role)...")
        return {
            'text':  TextEmotionClassifier(),
            'audio': AudioEmotionClassifier(),
            'role':  ProductionSpeakerRoleClassifier(device=DEVICE),
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process(self, audio_path: str, language: str = None) -> List[Dict]:
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return []

        print(f"Processing: {audio_path}")
        start_time = time.time()

        # ============================================================
        # PHASE 1 — ASR + DIARIZATION
        # ============================================================
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

        # Cleanup Phase 1 models immediately
        print(">> Cleaning up ASR models...")
        del asr_model, diarize_model, model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Memory Freed")

        # ============================================================
        # PHASE 2 — ROLE + EMOTION ANALYSIS
        # ============================================================
        print(">> Step 5: Speaker Role Classification...")
        models         = self._load_analysis_models()
        role_clf       = models['role']
        text_clf       = models['text']
        audio_clf      = models['audio']

        # Build flat segment list for role classifier
        flat_segments = [
            {"text": seg["text"].strip(), "speaker": seg.get("speaker", "UNKNOWN")}
            for seg in result["segments"]
        ]
        speaker_roles = role_clf.classify_all_speakers(flat_segments)

        # ── free role classifier (BART-large) before emotion inference ──
        del role_clf, models['role']
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("     [OK] Role classifier freed")

        # ============================================================
        # PHASE 3 — EMOTION (batched text + sequential audio)
        # ============================================================
        print("\n>> Step 6: Analyzing Emotions...")

        # --- Batch all text-emotion predictions in one forward pass ---
        all_texts       = [seg["text"].strip() for seg in result["segments"]]
        text_predictions= text_clf.predict_batch(all_texts)

        # --- Load audio once, slice per segment ---
        full_audio, _sr = librosa.load(audio_path, sr=16000)

        final_segments = []
        for idx, seg in enumerate(result["segments"]):
            start   = seg["start"]
            end     = seg["end"]
            text    = seg["text"].strip()
            speaker = seg.get("speaker", "UNKNOWN")
            role    = speaker_roles.get(speaker, "UNKNOWN")

            # Audio slice
            audio_seg = full_audio[int(start * 16000):int(end * 16000)]
            audio_res = audio_clf.predict(audio_seg)

            # Text result already computed in batch
            text_res  = text_predictions[idx]

            # Fusion
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

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    @staticmethod
    def _print_utterance(speaker, role, text, start, end, text_res, audio_res, fusion, is_overlap=False):
        overlap_tag = " [OVERLAP]" if is_overlap else ""

        t_emo, t_conf = text_res['emotion'],  text_res['confidence']
        a_emo, a_conf = audio_res['emotion'], audio_res['confidence']
        dims          = audio_res.get('dimensions', {})
        dims_str      = (f" [V:{dims.get('valence',0):.2f} "
                         f"A:{dims.get('arousal',0):.2f} "
                         f"D:{dims.get('dominance',0):.2f}]") if dims else ""

        f_emo, f_conf, f_src = fusion['emotion'], fusion['confidence'], fusion['source']
        corrected_str = (f" (was: {fusion['original_emotion']})"
                         if fusion.get('corrected') and fusion.get('original_emotion') else "")

        print(f"\n[{start:.1f}s-{end:.1f}s] {role}{overlap_tag}")
        print(f"   Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"   Text Emotion:  {t_emo:12s} ({t_conf:5.1%}){corrected_str}")
        print(f"   Audio Emotion: {a_emo:12s} ({a_conf:5.1%}){dims_str}")
        print(f"   >> Fused:      {f_emo:12s} ({f_conf:5.1%}) [{f_src}]")


# ==============================================================================
# JSON SERIALIZATION HELPER
# ==============================================================================

def save_results_json(results: List[Dict], output_path: str):
    def _convert(obj):
        if isinstance(obj, dict):  return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_convert(v) for v in obj]
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
    parser = argparse.ArgumentParser(description="VocalMind Production Pipeline v2 (Optimized)")
    parser.add_argument("--audio",    type=str, default=DEFAULT_AUDIO_FILE, help="Input audio file")
    parser.add_argument("--language", type=str, default="en",               help="Language code")
    parser.add_argument("--output",   type=str,                             help="Save results to JSON")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)

    pipeline = CombinedPipeline()
    results  = pipeline.process(args.audio, language=args.language)

    if args.output and results:
        save_results_json(results, args.output)

    print("\n" + "=" * 50)
    print("DONE.")