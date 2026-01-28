"""
HERB dataset specific ingestion strategy.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from llama_index.core import Document
from rag_app.config import settings
from rag_app.ingestion.base import BaseIngestionStrategy

logger = logging.getLogger(__name__)


class HERBDataLoader:
    """Loader for HERB enterprise dataset structure."""

    def __init__(self, data_root: Path):
        self.products_dir = data_root / "products"
        self.metadata_dir = data_root / "metadata"
        self._employees: Dict[str, dict] = {}
        self._customers: Dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata reference files."""
        try:
            emp_file = self.metadata_dir / "employee.json"
            if emp_file.exists():
                with open(emp_file, "r", encoding="utf-8") as f:
                    self._employees = {e.get("id"): e for e in json.load(f)}

            cust_file = self.metadata_dir / "customers_data.json"
            if cust_file.exists():
                with open(cust_file, "r", encoding="utf-8") as f:
                    self._customers = {c.get("id"): c for c in json.load(f)}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    def _resolve_employee(self, eid: str) -> str:
        return self._employees.get(eid, {}).get("name", eid)

    def _parse_slack(self, msg: dict) -> str:
        user_info = msg.get("Message", {}).get("User", {})
        sender = self._resolve_employee(user_info.get("userId", "unknown"))
        text = user_info.get("text", "")
        
        replies = []
        for r in msg.get("ThreadReplies", []):
            r_user = r.get("User", {})
            r_name = self._resolve_employee(r_user.get("userId", "unknown"))
            replies.append(f"  - {r_name}: {r_user.get('text', '')}")
            
        return f"[Slack] {sender}: {text}\nReplies:\n" + "\n".join(replies) if replies else f"[Slack] {sender}: {text}"

    def _parse_meeting(self, meeting: dict) -> str:
        if "transcript" in meeting:
            return f"[Meeting Transcript]\n{meeting['transcript']}"
        
        messages = meeting.get("messages", [])
        if messages:
            chat = [f"{self._resolve_employee(m.get('sender', 'unknown'))}: {m.get('text', '')}" for m in messages]
            return "[Meeting Chat]\n" + "\n".join(chat)
            
        return f"[Meeting Data] {json.dumps(meeting)}"

    def _parse_doc(self, doc: dict) -> str:
        return f"[Document: {doc.get('title', 'Untitled')}]\n{doc.get('content', '')}"

    def _parse_pr(self, pr: dict) -> str:
        author = self._resolve_employee(pr.get("author", "unknown"))
        comments = [
            f"  - {self._resolve_employee(c.get('author'))}: {c.get('body')}" 
            for c in pr.get("comments", [])
        ]
        return (
            f"[PR: {pr.get('title')}]\nAuthor: {author}\n{pr.get('description', '')}\n" + 
            ("Comments:\n" + "\n".join(comments) if comments else "")
        )

    def load_product(self, product_file: Path) -> List[Document]:
        try:
            with open(product_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {product_file}: {e}")
            return []

        documents = []
        product_name = product_file.stem
        
        # Schema mapping: source_key -> (parser_func, doc_type)
        parsers: Dict[str, tuple] = {
            "slack": (self._parse_slack, "slack"),
            "meeting_transcripts": (self._parse_meeting, "meeting"),
            "meeting_chats": (self._parse_meeting, "meeting"),
            "meetings": (self._parse_meeting, "meeting"),
            "documents": (self._parse_doc, "document"),
            "pull_requests": (self._parse_pr, "pull_request"),
        }

        for key, (func, doc_type) in parsers.items():
            for item in data.get(key, []):
                try:
                    content = func(item)
                    documents.append(Document(
                        text=content,
                        metadata={
                            "source": "herb",
                            "product": product_name,
                            "type": doc_type,
                            "id": item.get("id", "unknown"),
                            "timestamp": item.get("timestamp", "")
                        }
                    ))
                except Exception as e:
                    logger.debug(f"Skipping item in {product_name}/{key}: {e}")

        return documents

    def load_all(self) -> List[Document]:
        if not self.products_dir.exists():
            print(f"Warning: HERB products directory {self.products_dir} not found.")
            return []

        all_docs = []
        for p_file in self.products_dir.glob("*.json"):
            docs = self.load_product(p_file)
            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} docs from {p_file.name}")
            
        return all_docs


class HERBIngestionStrategy(BaseIngestionStrategy):
    """Ingests data from specific HERB dataset structure."""
    
    def load_documents(self) -> List[Document]:
        print(f"Loading HERB dataset from: {settings.DATA_DIR}")
        return HERBDataLoader(settings.DATA_DIR).load_all()
