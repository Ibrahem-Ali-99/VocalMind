import json
import os

import requests


BASE_URL = "https://etta-cleistogamous-untangentially.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}
FILE_PATH = r"g:\projects\VocalMind\research\voice-gen\telecom_call.mp3"


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
            files = {"file": (filename, handle, "audio/mpeg")}
            response = requests.post(url, headers=HEADERS, files=files)
            print(f"Status: {response.status_code}")
            body = response.text
            if len(body) > 1000:
                print(f"Body (truncated): {body[:1000]}...\n")
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

    print(f"Testing all endpoints with: {os.path.basename(FILE_PATH)}")
    results["transcribe"] = test_post("/transcribe", FILE_PATH)
    results["emotion"] = test_post("/emotion", FILE_PATH)
    results["diarize"] = test_post("/diarize", FILE_PATH)
    results["vad"] = test_post("/vad", FILE_PATH)
    results["full"] = test_post("/full", FILE_PATH)

    with open("telecom_test_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print("Test complete. Results saved to telecom_test_results.json")
