from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx


DEFAULT_BASE_URL = "https://etta-cleistogamous-untangentially.ngrok-free.dev"
DEFAULT_AUDIO_FILE = Path("AudioData/nexalink/sample.wav")
HEADERS = {"ngrok-skip-browser-warning": "true"}


async def _probe_endpoint(client: httpx.AsyncClient, base_url: str, endpoint: str, payload: bytes) -> None:
    print(f"\n=== {endpoint.upper()} ===")
    try:
        response = await client.post(
            f"{base_url}/{endpoint}",
            files={"file": ("smoke_test.wav", payload, "audio/wav")},
            headers=HEADERS,
        )
        print(f"Status: {response.status_code}")
        print(f"Body (first 300 chars): {response.text[:300]}...")
    except Exception as exc:  # pragma: no cover - network script
        print(f"Request failed: {exc}")


async def run_smoke_test(base_url: str, audio_file: Path, timeout_s: int) -> None:
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    audio_bytes = audio_file.read_bytes()
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        await _probe_endpoint(client, base_url.rstrip("/"), "predict", audio_bytes)
        await _probe_endpoint(client, base_url.rstrip("/"), "split", audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic smoke checks against Kaggle inference endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Kaggle/ngrok base URL.")
    parser.add_argument("--audio-file", type=Path, default=DEFAULT_AUDIO_FILE, help="Path to sample audio file.")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds.")
    args = parser.parse_args()

    asyncio.run(run_smoke_test(args.base_url, args.audio_file, args.timeout))


if __name__ == "__main__":
    main()
