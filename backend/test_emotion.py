import asyncio
from httpx import AsyncClient
import os
import pytest

@pytest.mark.asyncio
async def test_backend_proxy():
    backend_url = "http://localhost:8001/api/v1/emotion/analyze"
    audio_file = r"g:\projects\VocalMind\test_short.wav"

    if not os.path.exists(audio_file):
        print(f"Error: Could not find test audio file at {audio_file}")
        return

    print(f"Testing Backend Proxy Endpoint: {backend_url}")
    print("This will send the file to the backend, which proxies to the GPU container...")

    try:
        async with AsyncClient(timeout=65.0) as client:
            with open(audio_file, "rb") as f:
                response = await client.post(
                    backend_url,
                    files={"file": (os.path.basename(audio_file), f, "audio/wav")}
                )

            print(f"\nBackend responded with Status: {response.status_code}")
            try:
                print("JSON Data:", response.json())
            except Exception:
                print("Text Data:", response.text)

    except Exception as e:
        print(f"Proxy request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_backend_proxy())
