# kaggle_worker.py
# Run this entire script in a Kaggle Notebook cell.

"""
!pip install funasr modelscope torchaudio librosa soundfile fastapi uvicorn pyngrok python-multipart nest-asyncio
"""

import os
import uuid
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from pyngrok import ngrok
from funasr import AutoModel

# --- CONFIGURATION ---
# This is the shared secret key. Must match KAGGLE_API_SECRET in your backend .env
SHARED_SECRET = "vocalmind_secret_gpu_key"

# 1. Initialize FastAPI
app = FastAPI(title="VocalMind Emotion API (Hardened)")

# 2. Load the model globally at startup
print("Loading emotion2vec_plus_large model...")
model = AutoModel(
    model="iic/emotion2vec_plus_large",
    hub="ms",  # modelscope
    disable_update=True,
    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
)
print("Model loaded successfully!")

# 3. Define the /analyze endpoint
@app.post("/analyze")
async def analyze_emotion(
    file: UploadFile = File(...), 
    x_api_key: str = Header(None, alias="X-API-Key")
):
    # Security Check
    if x_api_key != SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

    # Generate a unique filename using UUID to prevent collisions
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}_{file.filename}"
    
    try:
        content = await file.read()
        if not content:
             raise HTTPException(status_code=400, detail="Empty file uploaded")
             
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Run inference
        res = model.generate(temp_path, output_dir="./outputs", granularity="utterance")
        
        if not res:
             raise HTTPException(status_code=500, detail="Inference returned no results")

        result_dict = res[0]
        labels = result_dict.get("labels", [])
        scores = result_dict.get("scores", [])
        
        # Clean labels (e.g. "开心/happy" -> "happy")
        cleaned_labels = [label.split("/")[-1] for label in labels]
        
        # Build emotions list
        emotions = []
        for label, score in zip(cleaned_labels, scores):
            emotions.append({"label": label, "score": float(score)})
            
        # Sort by score descending
        emotions.sort(key=lambda x: x["score"], reverse=True)
        
        top_emotion = emotions[0]["label"] if emotions else None
        top_score = emotions[0]["score"] if emotions else None
        
        return {
            "top_emotion": top_emotion,
            "top_score": top_score,
            "emotions": emotions,
            "filename": file.filename
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 4. Start ngrok and uvicorn
if __name__ == "__main__":
    import threading
    import time
    
    # Important for running uvicorn in a Jupyter/Kaggle notebook environment
    nest_asyncio.apply()
    
    # Optional: Set your authtoken if you want persistent tunnels:
    # ngrok.set_auth_token("YOUR_TOKEN_HERE")

    port = 8000
    
    # Connect ngrok
    tunnels = ngrok.get_tunnels()
    for t in tunnels:
        ngrok.disconnect(t.public_url)
        
    public_url = ngrok.connect(port).public_url
    print(f"\n{'='*60}")
    print(f"✅ Kaggle Worker is SECURED and RUNNING")
    print(f"🚀 PUBLIC URL: {public_url}")
    print(f"🔑 SHARED SECRET: {SHARED_SECRET}")
    print(f"Paste the URL above into your backend .env as KAGGLE_NGROK_URL")
    print(f"{'='*60}\n")
    
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    
    # Run uvicorn in a separate thread
    threading.Thread(target=run_server, daemon=True).start()
    
    # Keep the notebook cell alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping ngrok and server...")
        ngrok.kill()
