import asyncio
import os
import sys

# Ensure the app can be imported
sys.path.append(os.getcwd())

from app.api.routes.diarization.service import diarization_client
from app.core.config import settings

async def test_files(file_paths):
    print(f"Testing {len(file_paths)} files against: {settings.KAGGLE_SERVER_URL}")
    print(f"IS_LOCAL: {settings.IS_LOCAL}")
    
    for path in file_paths:
        print(f"\n--- Testing file: {os.path.basename(path)} ---")
        try:
            result = await diarization_client.analyze_local_file(path)
            print("Success!")
            # Print a summary of segments
            segments = result.get("segments", [])
            print(f"Found {len(segments)} segments.")
            if segments:
                for i, seg in enumerate(segments[:5]):
                    print(f"  {i}: {seg['start']} - {seg['end']} | Speaker: {seg['speaker']}")
                if len(segments) > 5:
                    print(f"  ... and {len(segments)-5} more.")
        except Exception as e:
            print(f"Error testing file: {e}")

if __name__ == "__main__":
    files = [
        r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077589677\2077589677_final_stereo.wav",
        r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077592167\2077592167_final_stereo.wav"
    ]
    asyncio.run(test_files(files))
