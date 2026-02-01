"""
VocalMind Speech Pipeline with Emotion Recognition
==================================================
Production-ready speech processing with multimodal emotion analysis:
- Speaker diarization (Pyannote 3.1)
- Speech recognition (faster-whisper)
- Text emotion (j-hartmann/emotion-english-distilroberta-base)
- Audio emotion (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
- Multimodal fusion with confidence weighting
- Business insights: stress reduction, empathy scoring, resolution tracking

Usage:
    python emotion_analysis.py --audio path/to/call.mp3
    python emotion_analysis.py --audio call.mp3 --output results.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import librosa
import numpy as np
import transformers

# Suppress all warnings including transformers logging issues
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable transformers warnings about sequential pipeline usage
transformers.logging.set_verbosity_error()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from speech_pipeline import VocalMindPipelineV2

# -----------------------------------------------------------------------------
# EMOTION MODELS
# -----------------------------------------------------------------------------

TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


# audeering model trained on MSP-Podcast dataset (natural conversational speech)
# This model outputs dimensional values: arousal, dominance, valence (0-1 scale)
AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

class AudioEmotionClassifier:
    """
    Audio-based emotion classifier using audeering's wav2vec2 model.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading audio emotion model: {AUDIO_EMOTION_MODEL}")
        
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(AUDIO_EMOTION_MODEL)
        # Use custom EmotionModel wrapper or load directly if simple classification
        # The audeering model is a fine-tuned wav2vec2 with a regression head
        from transformers import Wav2Vec2Model
        self.model = EmotionModel.from_pretrained(AUDIO_EMOTION_MODEL)
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        
        print(f"Audio emotion model loaded. Device: {self.device.upper()}")
    
    def _dimensions_to_emotion(self, arousal: float, valence: float, dominance: float) -> tuple:
        """Map dimensions to emotion categories."""
        v_low, v_high = 0.4, 0.6
        a_low, a_high = 0.4, 0.6
        
        if arousal > 0.7:
            if valence > 0.5: return "joy", valence * 0.9 + arousal * 0.1
            else: return ("anger" if dominance > 0.5 else "fear"), (1 - valence) * 0.8 + arousal * 0.2
        
        if valence > v_high:
            return ("joy" if arousal > 0.5 else "neutral"), valence
            
        if valence < v_low:
            if arousal > a_high: return ("anger" if dominance > 0.5 else "fear"), 1 - valence
            elif arousal < a_low: return "sadness", 1 - valence
            else: return "disgust", 1 - valence
            
        return "neutral", 0.5 + abs(valence - 0.5)

    def predict_segment(self, audio_array: np.ndarray, sr: int = 16000) -> Dict:
        if len(audio_array) < sr * 1.0:
            audio_array = np.pad(audio_array, (0, int(sr * 1.0) - len(audio_array)), mode='constant')
            
        y = self.processor(audio_array, sampling_rate=sr)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)
        
        with torch.no_grad():
            hidden_states, logits = self.model(y)
            dims = logits[0].cpu().numpy()
            
        arousal, dominance, valence = float(dims[0]), float(dims[1]), float(dims[2])
        emotion, conf = self._dimensions_to_emotion(arousal, valence, dominance)
        
        # Calculate scores for all 7 emotions based on dimensions
        all_scores = self._calculate_emotion_scores(arousal, valence, dominance)
        
        return {
            'emotion': emotion,
            'confidence': conf,
            'sentiment': AUDIO_TO_SENTIMENT.get(emotion, 'Neutral'),
            'dimensions': {'arousal': arousal, 'dominance': dominance, 'valence': valence},
            'all_scores': all_scores
        }
    
    def _calculate_emotion_scores(self, arousal: float, valence: float, dominance: float) -> Dict:
        """
        Calculate approximate scores for all 7 emotions based on dimensional values.
        Uses the circumplex model relationships.
        """
        def gaussian_score(v, a, v_center, a_center, sigma=0.3):
            dist = ((v - v_center)**2 + (a - a_center)**2) ** 0.5
            return max(0, 1 - dist / sigma)
        
        scores = {
            'anger': gaussian_score(valence, arousal, 0.2, 0.8) * (0.5 + dominance * 0.5),
            'disgust': gaussian_score(valence, arousal, 0.3, 0.5),
            'fear': gaussian_score(valence, arousal, 0.3, 0.8) * (1.5 - dominance),
            'joy': gaussian_score(valence, arousal, 0.8, 0.7),
            'neutral': gaussian_score(valence, arousal, 0.5, 0.5),
            'sadness': gaussian_score(valence, arousal, 0.2, 0.2),
            'surprise': gaussian_score(valence, arousal, 0.6, 0.9)
        }
        
        # Normalize scores
        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}

# Emoji mappings for 7 emotion classes (used by both text and audio models)
EMOTION_EMOJIS = {
    "anger": "🤬",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😀",
    "neutral": "😐",
    "sadness": "😭",
    "surprise": "😲",
    "uncertain": "❓",  # For low-confidence predictions
}

# Map emotions to simplified 3-class sentiment (same for text and audio)
EMOTION_TO_SENTIMENT = {
    "anger": "Negative",
    "disgust": "Negative",
    "fear": "Negative",
    "sadness": "Negative",
    "joy": "Positive",
    "surprise": "Neutral",  # Surprise is context-dependent, not always positive
    "neutral": "Neutral",
    "uncertain": "Neutral"
}

# Aliases for backward compatibility
TEXT_TO_SENTIMENT = EMOTION_TO_SENTIMENT
AUDIO_TO_SENTIMENT = EMOTION_TO_SENTIMENT

# -----------------------------------------------------------------------------
# EMOTION CORRECTION RULES
# -----------------------------------------------------------------------------

# Phrases that indicate positive/neutral intent
POSITIVE_PHRASE_PATTERNS = [
    # Gratitude
    "thank you", "thanks", "appreciate", "grateful", "thank you so much",
    # Farewells
    "have a nice day", "have a great day", "have a wonderful day",
    "take care", "good bye", "goodbye", "have a good", "bye bye",
    # Courtesy
    "welcome", "you're welcome", "my pleasure", "certainly", "absolutely",
    # Help offers
    "glad to help", "happy to help", "happy to assist", "here to help",
    "is there anything else", "can i help you with anything", "anything else i can do",
    "let me check", "i'll look into", "one moment please", "bear with me",
    # Empathy (agent)
    "i understand", "i apologize", "sorry about that", "sorry for",
    "no problem", "no worries", "of course", "right away",
    # Confirmations
    "sounds good", "perfect", "great", "wonderful", "excellent",
    "all set", "all done", "that's it", "you're all set",
]

# Phrases that indicate negative emotions (should not be joy/surprise positive)
NEGATIVE_PHRASE_PATTERNS = [
    "upset", "frustrated", "angry", "furious", "annoyed",
    "disappointed", "unhappy", "not happy", "terrible",
    "worst", "horrible", "ridiculous", "unacceptable",
    "fed up", "sick and tired", "can't believe", "this is insane",
]

# Confidence threshold - below this, label as "uncertain"
CONFIDENCE_THRESHOLD = 0.50

# Feature log for future model training
FEATURE_LOG: List[Dict] = []


import re

def normalize_text(text: str) -> str:
    """Normalize transcription artifacts: numbers, currencies, phone numbers."""
    normalized = text
    
    # Fix broken decimals: "8 .99" or "8. 99" → "8.99"
    normalized = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', normalized)
    
    # Fix broken currency: "$ 50" → "$50", "50 $" → "$50"
    normalized = re.sub(r'\$\s*(\d)', r'$\1', normalized)
    normalized = re.sub(r'(\d)\s*\$', r'$\1', normalized)
    
    # Normalize phone number fragments: rejoin broken chunks
    # "1 800 555 1234" or "1-800-555-1234" → keep as-is but fix "1 8 0 0" patterns
    normalized = re.sub(r'(\d)\s+(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', normalized)
    
    # Fix common ASR artifacts
    normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces
    normalized = normalized.strip()
    
    return normalized


def apply_emotion_correction(text: str, emotion: str, confidence: float) -> tuple:
    """
    Apply rule-based corrections for obvious mislabels.
    
    Returns:
        (corrected_emotion, corrected_confidence, was_corrected)
    """
    text_lower = text.lower()
    
    # Check for positive phrases being mislabeled as negative emotions
    if emotion in ['fear', 'anger', 'disgust', 'sadness']:
        for phrase in POSITIVE_PHRASE_PATTERNS:
            if phrase in text_lower:
                # Override to neutral (polite phrases are neutral, not negative)
                return 'neutral', max(confidence, 0.7), True
    
    # Check for negative phrases being mislabeled as positive emotions
    if emotion in ['joy', 'surprise']:
        for phrase in NEGATIVE_PHRASE_PATTERNS:
            if phrase in text_lower:
                # Check context - "I understand your frustration" is empathetic, not angry
                if "understand" in text_lower or "sorry" in text_lower:
                    return 'neutral', max(confidence, 0.6), True
                # Otherwise, actually negative - override to anger
                return 'anger', max(confidence, 0.75), True
    
    # Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return 'uncertain', confidence, True
    
    return emotion, confidence, False


class TextEmotionClassifier:
    """Text-based emotion classifier using DistilRoBERTa."""
    
    def __init__(self, device: Optional[str] = None):
        from transformers import pipeline
        
        self.device = 0 if (device == "cuda" or (device is None and torch.cuda.is_available())) else -1
        
        print(f"Loading text emotion model: {TEXT_EMOTION_MODEL}")
        self.classifier = pipeline(
            "text-classification",
            model=TEXT_EMOTION_MODEL,
            device=self.device,
            top_k=None
        )
        print(f"Text emotion model loaded. Device: {'CUDA' if self.device == 0 else 'CPU'}")
    
    def predict(self, text: str, apply_corrections: bool = True) -> Dict:
        """Predict emotion from text."""
        if not text.strip():
            return {'emotion': 'neutral', 'confidence': 0.0, 'sentiment': 'Neutral', 'corrected': False}
        
        # Normalize text before processing
        normalized_text = normalize_text(text)
        result = self.classifier(normalized_text)
        top = max(result[0], key=lambda x: x['score'])
        
        emotion = top['label']
        confidence = top['score']
        corrected = False
        
        # Apply rule-based corrections
        if apply_corrections:
            emotion, confidence, corrected = apply_emotion_correction(text, emotion, confidence)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'sentiment': TEXT_TO_SENTIMENT.get(emotion, 'Neutral'),
            'all_scores': {r['label']: r['score'] for r in result[0]},
            'corrected': corrected,
            'original_emotion': top['label'] if corrected else None
        }


class RegressionHead(torch.nn.Module):
    """Classification/Regression head for audeering emotion model."""
    
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
    """Speech emotion classifier with regression head for dimensional output."""
    
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




# -----------------------------------------------------------------------------
# FULL PIPELINE TEST
# -----------------------------------------------------------------------------

def apply_sentiment_smoothing(utterances: List[Dict], window_size: int = 3) -> List[Dict]:
    """
    Apply rolling average smoothing to reduce sentiment flip-flopping.
    Uses a window of adjacent utterances to stabilize predictions.
    """
    if len(utterances) < window_size:
        return utterances
    
    sentiment_values = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    value_to_sentiment = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    
    for i in range(len(utterances)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(utterances), i + window_size // 2 + 1)
        
        window = utterances[start_idx:end_idx]
        values = [sentiment_values.get(u['fused_emotion']['sentiment'], 0) for u in window]
        avg_value = sum(values) / len(values)
        
        # Map back to sentiment
        if avg_value > 0.3:
            smoothed = 'Positive'
        elif avg_value < -0.3:
            smoothed = 'Negative'
        else:
            smoothed = 'Neutral'
        
        utterances[i]['smoothed_sentiment'] = smoothed
    
    return utterances


def log_features(utterance: Dict, text_result: Dict, audio_result: Dict):
    """Log per-utterance features for future model training."""
    dims = audio_result.get('dimensions', {})
    
    feature_record = {
        'text_conf': text_result['confidence'],
        'audio_conf': audio_result['confidence'],
        'text_top_label': text_result['emotion'],
        'audio_top_label': audio_result['emotion'],
        'valence': dims.get('valence', 0.5),
        'arousal': dims.get('arousal', 0.5),
        'dominance': dims.get('dominance', 0.5),
        'utterance_length': len(utterance['text']),
        'speaker_role': utterance['role'],
        'fused_label': utterance.get('fused_emotion', {}).get('emotion', 'unknown'),
        'timestamp': utterance['timestamp'],
    }
    FEATURE_LOG.append(feature_record)


class SpeechPipelineWithEmotion:
    """
    Complete speech pipeline with emotion recognition.
    Combines VocalMindPipelineV2 (diarization + ASR),
    text emotion classification, and audio emotion classification.
    """
    
    def __init__(self, whisper_model: str = "medium"):
        self.speech_pipeline = VocalMindPipelineV2(whisper_model=whisper_model)
        self.text_emotion = TextEmotionClassifier()
        self.audio_emotion = AudioEmotionClassifier()
        print("All models loaded.\n")
    
    def process_file(self, audio_path: str) -> Dict:
        """
        Process audio file through the complete pipeline.
        
        Returns:
            Dictionary with conversation log and emotion analysis
        """
        start_time = time.time()
        
        print("\nRunning diarization + ASR...")
        conversation_log = self.speech_pipeline.process_file(audio_path)
        
        if not conversation_log:
            print("No conversation detected in audio.")
            return {'conversation': [], 'summary': {}}
        
        print("\nAnalyzing emotions...")
        full_audio, sr = librosa.load(audio_path, sr=16000)
        enhanced_log = []
        
        for idx, utterance in enumerate(conversation_log):
            text = utterance['text']
            start_sec = utterance['timestamp'][0]
            end_sec = utterance['timestamp'][1]
            role = utterance['role']
            
            # Extract audio segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            audio_segment = full_audio[start_sample:end_sample]
            
            # Text emotion
            text_result = self.text_emotion.predict(text)
            
            # Audio emotion
            audio_result = self.audio_emotion.predict_segment(audio_segment, sr)
            
            # Multimodal fusion: weighted combination of text and audio
            text_conf = text_result['confidence']
            audio_conf = audio_result['confidence']
            
            # Check if both agree on sentiment
            text_sent = text_result['sentiment']
            audio_sent = audio_result['sentiment']
            agreement = (text_sent == audio_sent)
            
            # Weight calculation: higher confidence = more weight
            # But cap audio weight when it's likely wrong (very generic predictions)
            text_weight = text_conf
            audio_weight = audio_conf * 0.8
            
            if agreement:
                fused_emotion = text_result['emotion'] if text_conf >= audio_conf else audio_result['emotion']
                fused_sentiment = text_sent
                fused_confidence = min(1.0, (text_conf + audio_conf) / 1.5)
                fusion_source = 'both-agree'
            else:
                # Special case: audio detected strong negative but text didn't
                # Trust audio if confidence is high (>75%)
                if audio_sent == 'Negative' and text_sent != 'Negative' and audio_conf > 0.75:
                    fused_emotion = audio_result['emotion']
                    fused_sentiment = audio_sent
                    fused_confidence = audio_conf
                    fusion_source = 'audio-override'
                elif text_weight > audio_weight:
                    fused_emotion = text_result['emotion']
                    fused_sentiment = text_sent
                    fused_confidence = text_conf
                    fusion_source = 'text'
                else:
                    fused_emotion = audio_result['emotion']
                    fused_sentiment = audio_sent
                    fused_confidence = audio_conf
                    fusion_source = 'audio'
            
            enhanced_utterance = {
                **utterance,
                'text_emotion': {
                    'emotion': text_result['emotion'],
                    'confidence': text_result['confidence'],
                    'sentiment': text_result['sentiment']
                },
                'audio_emotion': {
                    'emotion': audio_result['emotion'],
                    'confidence': audio_result['confidence'],
                    'sentiment': audio_result['sentiment'],
                    'dimensions': audio_result.get('dimensions', {})
                },
                'fused_emotion': {
                    'emotion': fused_emotion,
                    'sentiment': fused_sentiment,
                    'confidence': fused_confidence,
                    'source': fusion_source
                }
            }
            
            enhanced_log.append(enhanced_utterance)
            
            # Log features for model training
            log_features(enhanced_utterance, text_result, audio_result)
            
            # Display
            text_emoji = EMOTION_EMOJIS.get(text_result['emotion'], "❓")
            audio_emoji = EMOTION_EMOJIS.get(audio_result['emotion'], "❓")
            fused_emoji = EMOTION_EMOJIS.get(fused_emotion, "❓")
            
            # Get audio dimensions if available
            dims = audio_result.get('dimensions', {})
            dims_str = ""
            if dims:
                dims_str = f" [V:{dims.get('valence', 0):.2f} A:{dims.get('arousal', 0):.2f} D:{dims.get('dominance', 0):.2f}]"
            
            # Check if text emotion was corrected
            corrected_str = ""
            if text_result.get('corrected') and text_result.get('original_emotion'):
                corrected_str = f" (was: {text_result['original_emotion']} → corrected)"
            
            print(f"\n[{idx+1}] {role} ({start_sec:.1f}s - {end_sec:.1f}s)")
            print(f"    Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            print(f"    Text Emotion:  {text_emoji} {text_result['emotion']:12s} ({text_result['confidence']:.1%}) → {text_result['sentiment']}{corrected_str}")
            print(f"    Audio Emotion: {audio_emoji} {audio_result['emotion']:12s} ({audio_result['confidence']:.1%}) → {audio_result['sentiment']}{dims_str}")
            
            # Show fusion decision
            if fusion_source == 'both-agree':
                print(f"    ⚡ Fused:       {fused_emoji} {fused_emotion:12s} ({fused_confidence:.1%}) → {fused_sentiment} [TEXT+AUDIO AGREE ✓]")
            elif fusion_source == 'audio-override':
                print(f"    ⚡ Fused:       {fused_emoji} {fused_emotion:12s} ({fused_confidence:.1%}) → {fused_sentiment} [AUDIO OVERRIDE ❗]")
            elif fusion_source == 'text':
                print(f"    ⚡ Fused:       {fused_emoji} {fused_emotion:12s} ({fused_confidence:.1%}) → {fused_sentiment} [TEXT wins]")
            else:
                print(f"    ⚡ Fused:       {fused_emoji} {fused_emotion:12s} ({fused_confidence:.1%}) → {fused_sentiment} [AUDIO wins]")
        
        enhanced_log = apply_sentiment_smoothing(enhanced_log, window_size=3)
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self._generate_summary(enhanced_log, time.time() - start_time)
        self._print_summary(summary)
        
        return {
            'conversation': enhanced_log,
            'summary': summary
        }
    
    def _generate_summary(self, conversation: List[Dict], total_time: float) -> Dict:
        """Generate analysis summary with business insights."""
        # Separate by role
        agent_utterances = [u for u in conversation if u['role'] == 'Agent']
        customer_utterances = [u for u in conversation if u['role'] == 'Customer']
        
        def count_sentiments(utterances: List[Dict], source: str = 'text_emotion') -> Dict:
            counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            for u in utterances:
                sentiment = u[source]['sentiment']
                counts[sentiment] = counts.get(sentiment, 0) + 1
            return counts
        
        def get_emotion_journey(utterances: List[Dict], source: str = 'text_emotion') -> List[str]:
            return [u[source]['sentiment'] for u in utterances]
        
        customer_journey = get_emotion_journey(customer_utterances, 'fused_emotion')
        
        return {
            'total_utterances': len(conversation),
            'agent_utterances': len(agent_utterances),
            'customer_utterances': len(customer_utterances),
            'agent_sentiment': count_sentiments(agent_utterances, 'fused_emotion'),
            'customer_sentiment': count_sentiments(customer_utterances, 'fused_emotion'),
            'customer_journey': customer_journey,
            'processing_time_sec': total_time,
        }
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary."""
        print(f"\nTotal Utterances: {summary['total_utterances']}")
        print(f"  Agent: {summary['agent_utterances']} | Customer: {summary['customer_utterances']}")
        
        print(f"\n--- Agent Sentiment ---")
        for sentiment, count in summary['agent_sentiment'].items():
            pct = count / max(summary['agent_utterances'], 1) * 100
            bar = "█" * int(pct / 10)
            print(f"  {sentiment:10s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print(f"\n--- Customer Sentiment ---")
        for sentiment, count in summary['customer_sentiment'].items():
            pct = count / max(summary['customer_utterances'], 1) * 100
            bar = "█" * int(pct / 10)
            print(f"  {sentiment:10s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print(f"\n--- Customer Journey ---")
        journey = summary['customer_journey']
        if journey:
            print(f"  {' → '.join(journey)}")
        
        print(f"\n--- Processing Time ---")
        print(f"  {summary['processing_time_sec']:.2f}s")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VocalMind Speech Pipeline Test with Emotion Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_pipeline_test.py --audio telecom_call.mp3
  python speech_pipeline_test.py --audio call.mp3 --whisper-model large-v3
  python speech_pipeline_test.py --audio call.mp3 --output results.json
        """
    )
    
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file to process")
    parser.add_argument("--whisper-model", type=str, default="medium",
                        choices=["base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--output", type=str,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = SpeechPipelineWithEmotion(whisper_model=args.whisper_model)
    results = pipeline.process_file(args.audio)
    
    # Save results if requested
    if args.output:
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
        
        # Include feature log for model training
        results['feature_log'] = FEATURE_LOG
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(deep_convert(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
        print(f"Feature log contains {len(FEATURE_LOG)} utterance records for model training")
    
    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
