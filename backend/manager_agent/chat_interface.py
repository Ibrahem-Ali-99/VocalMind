"""Chat interface placeholder."""
from typing import Any, Dict


class ChatInterface:
    def send(self, message: str) -> Dict[str, Any]:
        """Return dummy chat response."""
        return {"reply": f"Echo: {message}"}
