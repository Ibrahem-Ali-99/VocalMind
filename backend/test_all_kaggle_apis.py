import requests
import json
import os

BASE_URL = "https://etta-cleistogamous-untangentially.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}

FILES = [
    r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077589677\2077589677_final_stereo.wav",
    r"g:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077592167\2077592167_final_stereo.wav"
]

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
            files = {"file": (filename, f, "audio/wav")}
            response = requests.post(url, headers=HEADERS, files=files)
            print(f"Status: {response.status_code}")
            # Truncate output if too long
            body = response.text
            if len(body) > 500:
                print(f"Body (truncated): {body[:500]}...\n")
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
    
    # Test other endpoints with the first file
    file_path = FILES[0]
    results["transcribe"] = test_post("/transcribe", file_path)
    results["emotion"] = test_post("/emotion", file_path)
    results["diarize"] = test_post("/diarize", file_path)
    results["vad"] = test_post("/vad", file_path)
    results["full"] = test_post("/full", file_path)
    
    # Save results to a temporary file for inspection
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Test complete. Results saved to test_results.json")
