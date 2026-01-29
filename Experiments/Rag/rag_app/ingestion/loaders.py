"""
Custom data loaders for the RAG application.
"""

import json
from pathlib import Path
from typing import List
from datasets import load_from_disk

from llama_index.core import Document, SimpleDirectoryReader
from rag_app.config import settings


class CustomDirectoryLoader(SimpleDirectoryReader):
    """
    Directory loader that supports sidecar metadata files.

    If file is 'example.pdf', it looks for 'example.pdf.meta.json'.
    """

    def load_data(self) -> List[Document]:
        documents = super().load_data()

        # Filter out sidecar metadata files themselves so they aren't indexed as content
        documents = [
            doc
            for doc in documents
            if not doc.metadata.get("file_path", "").endswith(".meta.json")
        ]

        for doc in documents:
            # doc.metadata['file_path'] is automatically set by SimpleDirectoryReader
            file_path = Path(doc.metadata.get("file_path"))
            meta_path = file_path.with_name(f"{file_path.name}.meta.json")

            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        sidecar_meta = json.load(f)
                        doc.metadata.update(sidecar_meta)
                except Exception as e:
                    print(
                        f"Warning: Failed to load sidecar metadata for {file_path}: {e}"
                    )

            # Heuristic Title: Filename without extension
            if "title" not in doc.metadata:
                doc.metadata["title"] = file_path.stem.replace("_", " ").title()

        return documents


class RagBenchLoader:
    """Loader for the RagBench dataset (HuggingFace format)."""

    def load_documents(self) -> List[Document]:
        data_path = settings.DATA_DIR / "ragbench"
        print(f"Loading RagBench dataset from: {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(
                f"RagBench data not found at {data_path}. Please run download script first."
            )

        try:
            dataset = load_from_disk(str(data_path))
        except Exception as e:
            raise ValueError(f"Failed to load RagBench dataset: {e}")

        documents = []

        # Process 'train', 'test', 'validation' splits
        for split in dataset.keys():
            print(f"Processing split: {split}")
            for item in dataset[split]:
                content = item.get("context") or item.get("documents")

                if isinstance(content, list):
                    content = "\n\n".join(content)

                if not content:
                    continue

                # Heuristic Title from ID or Question
                dataset_id = item.get("id", str(hash(content))[:8])
                question = item.get("question", "")
                title = f"RagBench {dataset_id}"
                if question:
                    title = f"QA: {question[:50]}..."

                doc = Document(
                    text=str(content),
                    metadata={
                        "source": "ragbench",
                        "split": split,
                        "question": question,
                        "dataset_id": dataset_id,
                        "title": title,
                        "section_summary": title,
                    },
                )
                documents.append(doc)

        print(f"Loaded {len(documents)} documents from RagBench.")
        return documents
