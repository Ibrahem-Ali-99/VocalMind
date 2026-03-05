import requests
import time
import os
import sys

url = "http://localhost:8000/predict"
# Ensure the path is correct
audio_file_path = r"g:\projects\VocalMind\2077589677_final_stereo.wav"

if not os.path.exists(audio_file_path):
    print(f"File not found: {audio_file_path}")
    sys.exit(1)

print(f"Testing with file: {audio_file_path}")

try:
    start_time = time.time()
    with open(audio_file_path, "rb") as f:
        # the API expects the file to be passed in a form field named 'file'
        files = {"file": f}
        response = requests.post(url, files=files)
    end_time = time.time()

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("Response:", response.json())
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response:", response.text)
except Exception as e:
    print(f"An error occurred: {e}")
