from google.oauth2 import id_token
from google.auth.transport import requests
from app.core.config import settings
from app.schemas.user import GoogleUser
import logging

logger = logging.getLogger(__name__)

def verify_google_token(token: str) -> GoogleUser | None:
    try:
        id_info = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            settings.GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,
        )

        return GoogleUser(
            email=id_info['email'],
            name=id_info.get('name', ''),
            picture=id_info.get('picture')
        )
    except ValueError as e:
        logger.error(f"Invalid Google token: {e}")
        return None
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        return None
