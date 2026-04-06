# Backend

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installation

1. Install `uv`:

   ```bash
   pip install uv
   # or
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

### Running the Server

```bash
uv run uvicorn main:app --reload
```

### Development

- Add a package:
  ```bash
  uv add <package>
  ```
- Run tests (if any):
  ```bash
  uv run pytest
  ```

## Related Documentation

- `docs/explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md`
- `docs/llm_trigger/LLM_TRIGGER_FEATURE_GUIDE.md`
- `docs/rag/RAG_OVERVIEW.md`
