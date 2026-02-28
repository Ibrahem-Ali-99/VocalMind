import httpx
from fastapi import HTTPException, UploadFile
from typing import Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmotionAPIClient:
    """Client to interact with the standalone Emotion API Docker container."""
    
    def __init__(self):
        self.base_url = settings.EMOTION_API_URL.rstrip('/')
        self.predict_url = f"{self.base_url}/predict"
        
    async def analyze_audio(self, file: UploadFile) -> Dict[str, Any]:
        """
        Sends an audio file to the Emotion API container for analysis.
        Uses a long timeout because raw model inference on large clips can take time.
        """
        timeout = httpx.Timeout(60.0, connect=10.0)
        
        try:
            # Read the file content
            file_content = await file.read()
            # Seek back to 0 in case the file needs to be read again elsewhere
            await file.seek(0)
            
            files = {
                "file": (file.filename, file_content, file.content_type)
            }
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.predict_url, files=files)
                
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Emotion API returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Emotion analysis service error: {response.text}"
                )
                
        except httpx.RequestError as e:
            logger.error(f"Request to Emotion API failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Emotion analysis service is currently unreachable. Ensure the emotion-api-gpu container is running."
            )
        except httpx.TimeoutException:
            logger.error("Request to Emotion API timed out.")
            raise HTTPException(
                status_code=504,
                detail="Emotion analysis service timed out."
            )

# Singleton instance
emotion_client = EmotionAPIClient()
