"""Pre-download Silero VAD model so it's cached inside the Docker image."""
import torch

print("Downloading Silero VAD model...")
torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
print("Silero VAD model cached successfully.")
