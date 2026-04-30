import json
import os

import requests


BASE_URL = "https://etta-cleistogamous-untangentially.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}

FILES = [
    r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077589677\2077589677_final_stereo.wav",
    r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077592167\2077592167_final_stereo.wav",
]


def test_get(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing GET {url}...")
    try:
        response = requests.get(url, headers=HEADERS)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}\n")
        return response.json() if response.status_code == 200 else None
    except Exception as exc:
        print(f"Error: {exc}\n")
        return None


def test_post(endpoint: str, file_path: str):
    url = f"{BASE_URL}{endpoint}"
    filename = os.path.basename(file_path)
    print(f"Testing POST {url} with {filename}...")
    try:
        with open(file_path, "rb") as handle:
            files = {"file": (filename, handle, "audio/wav")}
            response = requests.post(url, headers=HEADERS, files=files)
            print(f"Status: {response.status_code}")
            body = response.text
            if len(body) > 500:
                print(f"Body (truncated): {body[:500]}...\n")
            else:
                print(f"Body: {body}\n")
            return response.json() if response.status_code == 200 else None
    except Exception as exc:
        print(f"Error: {exc}\n")
        return None


if __name__ == "__main__":
    results = {
        "health": test_get("/health"),
    }

    file_path = FILES[0]
    results["transcribe"] = test_post("/transcribe", file_path)
    results["emotion"] = test_post("/emotion", file_path)
    results["diarize"] = test_post("/diarize", file_path)
    results["vad"] = test_post("/vad", file_path)
    results["full"] = test_post("/full", file_path)

    with open("test_results.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print("Test complete. Results saved to test_results.json")
