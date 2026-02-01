"""
VocalMind Demo Script
=====================
Simple demo to test the speech pipeline with emotion recognition.

Usage:
    1. Edit AUDIO_FILE variable below to point to your audio file
    2. Run: python run_demo.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit this to test different audio files
# ============================================================================
AUDIO_FILE = "hard_overlap.mp3"  # Just filename - searches in generated_audio/
WHISPER_MODEL = "medium"  # Options: base, small, medium, large-v3
# ============================================================================


def run_pipeline(audio_file: str, whisper_model: str = "medium"):
    """Run the speech pipeline with emotion recognition on an audio file."""
    
    # Resolve path if only filename is provided
    if not os.path.isabs(audio_file) and not os.path.exists(audio_file):
        audio_dir = Path(__file__).parent.parent / "Voice-Generation" / "generated_audio"
        potential_path = audio_dir / audio_file
        if potential_path.exists():
            audio_file = str(potential_path)
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print(f"\n💡 Available files in generated_audio/:")
        audio_dir = Path(__file__).parent.parent / "Voice-Generation" / "generated_audio"
        if audio_dir.exists():
            for f in sorted(audio_dir.glob("*.mp3")):
                print(f"   - {f.name}")
        return
    
    print("\n" + "="*60)
    print("VOCALMIND PIPELINE DEMO")
    print("="*60)
    print(f"\n📁 Audio: {audio_file}")
    print(f"🎤 Whisper Model: {whisper_model}\n")
    
    try:
        from emotion_analysis import SpeechPipelineWithEmotion
        
        pipeline = SpeechPipelineWithEmotion(whisper_model=whisper_model)
        results = pipeline.process_file(audio_file)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
            
    except ImportError as e:
        print(f"❌ Error importing pipeline: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline(AUDIO_FILE, WHISPER_MODEL)
