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
                    data = json.load(f)
                    if isinstance(data, list):
                        self._employees = {e.get("id"): e for e in data}
                    elif isinstance(data, dict):
                        self._employees = data
                    else:
                        logger.warning(f"Unexpected format in {emp_file}")

            cust_file = self.metadata_dir / "customers_data.json"
            if cust_file.exists():
                with open(cust_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._customers = {c.get("id"): c for c in data}
                    else:
                        self._customers = data

        except Exception as e:
            logger.error(f"CRITICAL: Failed to load metadata: {e}")
            raise e

    def _resolve_employee(self, eid: str) -> str:
        name = self._employees.get(eid, {}).get("name", eid)
        if name != eid:
            return f"{name} ({eid})"
        return eid

    def _resolve_customer(self, cust_id: str) -> str:
        name = self._customers.get(cust_id, {}).get("name")
        if name:
            return f"{name} ({cust_id})"
        return cust_id

    def _enrich_text(self, text: str) -> str:
        if not text:
            return ""

        import re

        def replace_eid(match):
            eid = match.group(0)
            return self._resolve_employee(eid)

        text = re.sub(r"eid_[a-f0-9]+", replace_eid, text)

        def replace_cust(match):
            cid = match.group(0)
            return self._resolve_customer(cid)

        text = re.sub(r"CUST-\d+", replace_cust, text)

        return text

    # --- Parsing Methods with Heuristic Titles ---

    def _parse_slack(self, msg: dict) -> tuple[str, str]:
        user_info = msg.get("Message", {}).get("User", {})
        sender = self._resolve_employee(user_info.get("userId", "unknown"))
        text = self._enrich_text(user_info.get("text", ""))

        replies = []
        for r in msg.get("ThreadReplies", []):
            r_user = r.get("User", {})
            r_name = self._resolve_employee(r_user.get("userId", "unknown"))
            r_text = self._enrich_text(r_user.get("text", ""))
            replies.append(f"  - {r_name}: {r_text}")

        content = (
            f"[Slack] {sender}: {text}\nReplies:\n" + "\n".join(replies)
            if replies
            else f"[Slack] {sender}: {text}"
        )

        # Heuristic Title
        title = f"Slack Thread: {sender} - {text[:30]}..."
        return content, title

    def _parse_meeting(self, meeting: dict) -> tuple[str, str]:
        if "transcript" in meeting:
            content = (
                f"[Meeting Transcript]\n{self._enrich_text(meeting['transcript'])}"
            )
            title = f"Meeting Transcript - {meeting.get('id', 'unknown')}"
            return content, title

        messages = meeting.get("messages", [])
        if messages:
            chat = [
                f"{self._resolve_employee(m.get('sender', 'unknown'))}: {self._enrich_text(m.get('text', ''))}"
                for m in messages
            ]
            content = "[Meeting Chat]\n" + "\n".join(chat)
            title = f"Meeting Chat - {meeting.get('id', 'unknown')}"
            return content, title

        return f"[Meeting Data] {json.dumps(meeting)}", "Meeting Data"

    def _parse_doc(self, doc: dict) -> tuple[str, str]:
        doc_title = doc.get("title", "Untitled")
        content = (
            f"[Document: {doc_title}]\n{self._enrich_text(doc.get('content', ''))}"
        )
        return content, doc_title

    def _parse_pr(self, pr: dict) -> tuple[str, str]:
        author = self._resolve_employee(pr.get("author", "unknown"))
        desc = self._enrich_text(pr.get("description", ""))
        pr_title = pr.get("title", "Untitled PR")

        comments = [
            f"  - {self._resolve_employee(c.get('author'))}: {self._enrich_text(c.get('body'))}"
            for c in pr.get("comments", [])
        ]
        content = f"[PR: {pr_title}]\nAuthor: {author}\n{desc}\n" + (
            "Comments:\n" + "\n".join(comments) if comments else ""
        )
        return content, f"PR: {pr_title}"

    def load_product(self, product_file: Path) -> List[Document]:
        try:
            with open(product_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {product_file}: {e}")
            return []

        documents = []
        product_name = product_file.stem

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
                    content, title = func(item)

                    documents.append(
                        Document(
                            text=content,
                            metadata={
                                "source": "herb",
                                "product": product_name,
                                "type": doc_type,
                                "id": item.get("id", "unknown"),
                                "timestamp": item.get("timestamp", ""),
                                "title": title,  # Heuristic Title
                                "section_summary": title,  # Fallback summary
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"Skipping item in {product_name}/{key}: {e}")

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
