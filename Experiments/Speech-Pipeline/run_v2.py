
import sys
import os

# Ensure we can import main_v2
sys.path.append(os.path.dirname(__file__))

import main_v2

def run():
    print("Running Combined Pipeline via wrapper...")
    audio_path = "Experiments/Voice-Generation/telecom_call.mp3"
    
    # Run
    pipeline = main_v2.CombinedPipeline()
    results = pipeline.process(audio_path)
    
    # Save to file for verification
    with open("main_v2_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Processed {len(results)} segments.\n")
        f.write("-" * 50 + "\n")
        for seg in results:
            emo = seg['emotion_analysis']
            role = seg.get('role', 'UNKNOWN')
            f.write(f"{seg['speaker']} ({role}): {seg['text']}\n")
            f.write(f"  [{emo['emotion'].upper()}] ({emo['confidence']:.2f}) - {emo['sentiment']}\n")
            f.write("\n")
            
    print("Done. Saved to main_v2_results.txt")

if __name__ == "__main__":
    run()
