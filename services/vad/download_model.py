"""Pre-download Silero VAD model so it's cached inside the Docker image."""
from __future__ import annotations

import time


def main() -> None:
    import torch

    last: BaseException | None = None
    for attempt in range(1, 4):
        try:
            print(f"Downloading Silero VAD model (attempt {attempt}/3)...")
            torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            print("Silero VAD model cached successfully.")
            return
        except BaseException as exc:
            last = exc
            print(f"Attempt {attempt} failed: {exc!r}")
            if attempt < 3:
                time.sleep(15.0 * attempt)
    raise RuntimeError("Silero VAD model download failed after retries") from last


if __name__ == "__main__":
    main()
