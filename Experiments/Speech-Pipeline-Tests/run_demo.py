
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
from pathlib import Path

# Setup paths - this script is in Experiments/Speech-Pipeline-Tests/
current_dir = Path(__file__).resolve().parent
experiments_dir = current_dir.parent  # Experiments/
project_root = experiments_dir.parent  # VocalMind/

# Add Voice-Generation to sys.path
voice_gen_dir = experiments_dir / "Voice-Generation"
sys.path.append(str(voice_gen_dir))

def run_demo(model_path):
    print("=== VocalMind End-to-End Demo ===")
    
    # 1. Generate Call
    print("\n[Step 1] Generating Synthetic Call...")
    try:
        import voice_generation
        
        # Check if API Key is set, otherwise warn
        if not voice_generation.ELEVENLABS_API_KEY:
            print("WARNING: ELEVENLABS_API_KEY not found in .env.")
            print("If 'telecom_call.mp3' exists, we will use it. Otherwise, this will fail.")
        
        # Run generation
        # We call the main function or logic. 
        # voice_generation.py has generate_call().
        # It saves 'telecom_call.mp3' in CWD.
        potential_files = ["telecom_call.mp3", str(voice_gen_dir / "telecom_call.mp3")]
        existing_file = next((f for f in potential_files if os.path.exists(f)), None)
        
        if existing_file:
             print(f"Found existing '{existing_file}', skipping generation to save credits.")
             # Optional: Allow force regeneration via arg, but sticking to safe default
        else:
             if not voice_generation.ELEVENLABS_API_KEY:
                 print("Error: No API Key and no existing audio file.")
                 return
             voice_generation.generate_call()
             
    except ImportError as e:
        print(f"Error importing voice_generation: {e}")
        print("Please ensure 'elevenlabs', 'pydub', 'python-dotenv' are installed.")
        return
    except Exception as e:
        print(f"Error during voice generation: {e}")
        return

    audio_file = "telecom_call.mp3"
    if os.path.exists(audio_file):
        pass
    elif (voice_gen_dir / "telecom_call.mp3").exists():
        audio_file = str(voice_gen_dir / "telecom_call.mp3")
    else:
        print("Error: Audio file creation failed or file not found.")
        return

    # 2. Run Inference
    print(f"\n[Step 2] Running Inference Pipeline on {audio_file}...")
    try:
        # Use v2 pipeline with faster-whisper large-v3
        from inference_pipeline_v2 import VocalMindPipelineV2 as VocalMindPipeline
        
        if not os.path.exists(model_path):
             print(f"Model file not found at {model_path}")
             # Create a dummy model file if it's just a test structure? No, strictly require model.
             print("Please ensure you have downloaded 'best_model_3class.pt'.")
             return

        pipeline = VocalMindPipeline(model_path)
        conversation_log = pipeline.process_file(audio_file)
        
        print("\n[Step 3] Output:")
        for turn in conversation_log:
            print(f"{turn['role']} ({turn['emotion']}): {turn['text']}")
            
    except ImportError as e:
        print(f"Error importing pipeline: {e}")
        return
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full VocalMind demo")
    
    # Default model path relative to project structure
    default_model = str(experiments_dir / "Emotion-Recognition" / "best_model_3class.pt")
    
    parser.add_argument("--model", type=str, 
                        default=default_model,
                        help="Path to emotion model checkpoint")
    
    args = parser.parse_args()
    run_demo(args.model)
