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
