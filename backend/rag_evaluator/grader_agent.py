"""RAG grader agent placeholder."""
from typing import Dict, List


class GraderAgent:
    def grade(self, query: str, candidates: List[str]) -> Dict[str, float]:
        """Return dummy relevance scores for candidates."""
        return {candidate: 0.5 for candidate in candidates}
