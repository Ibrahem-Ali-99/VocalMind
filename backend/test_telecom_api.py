import requests
import json
import os

BASE_URL = "https://etta-cleistogamous-untangentially.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}

FILE_PATH = r"g:\projects\VocalMind\research\voice-gen\telecom_call.mp3"

def test_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing GET {url}...")
    try:
        response = requests.get(url, headers=HEADERS)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}\n")
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error: {e}\n")
        return None

def test_post(endpoint, file_path):
    url = f"{BASE_URL}{endpoint}"
    filename = os.path.basename(file_path)
    print(f"Testing POST {url} with {filename}...")
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "audio/mpeg")}
            # Note: We use audio/mpeg for mp3 files
            response = requests.post(url, headers=HEADERS, files=files)
            print(f"Status: {response.status_code}")
            body = response.text
            if len(body) > 1000:
                print(f"Body (truncated): {body[:1000]}...\n")
            else:
                print(f"Body: {body}\n")
            return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error: {e}\n")
        return None

if __name__ == "__main__":
    results = {}
    
    # Test Health
    results["health"] = test_get("/health")
    
    # Test all endpoints with the telecom call
    print(f"Testing all endpoints with: {os.path.basename(FILE_PATH)}")
    results["transcribe"] = test_post("/transcribe", FILE_PATH)
    results["emotion"] = test_post("/emotion", FILE_PATH)
    results["diarize"] = test_post("/diarize", FILE_PATH)
    results["vad"] = test_post("/vad", FILE_PATH)
    results["full"] = test_post("/full", FILE_PATH)
    
    with open("telecom_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Test complete. Results saved to telecom_test_results.json")
