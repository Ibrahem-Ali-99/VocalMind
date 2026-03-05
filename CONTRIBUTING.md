# Contributing to VocalMind

Thank you for your interest in contributing to VocalMind! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **[uv](https://github.com/astral-sh/uv)** (Python dependency management)
- **Docker & Docker Compose** (for infrastructure services)

### Local Setup

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-org/VocalMind.git
   cd VocalMind
   ```

2. **Start infrastructure services:**

   ```bash
   make up  # Starts DB, Ollama, Qdrant, VAD, etc.
   ```

3. **Backend:**

   ```bash
   make backend-install
   make backend-dev
   ```

4. **Frontend:**
   ```bash
   make frontend-install
   make frontend-dev
   ```

## 📁 Project Structure

| Directory   | Description                                 |
| :---------- | :------------------------------------------ |
| `backend/`  | FastAPI main API gateway                    |
| `frontend/` | Vite + React dashboard                      |
| `services/` | Microservices (VAD, WhisperX, Emotion, RAG) |
| `infra/`    | Infrastructure (DB init SQL, scripts)       |
| `research/` | Jupyter notebooks and experimental scripts  |
| `docs/`     | Project documentation                       |

## 🔀 Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Make changes** and commit with clear messages
4. **Run tests** before submitting:
   ```bash
   make backend-test
   make backend-lint
   make frontend-build
   ```
5. **Open a PR** against `main`

## 🧪 Running Tests

```bash
make backend-test    # Backend pytest suite
make backend-lint    # Ruff linter
make frontend-build  # TypeScript + Vite build check
```

## 📝 Coding Standards

- **Python**: Follow PEP 8, enforced by [Ruff](https://docs.astral.sh/ruff/)
- **TypeScript**: ESLint + Prettier
- **Commits**: Use conventional commit messages (e.g., `feat:`, `fix:`, `docs:`)

## 🐛 Reporting Bugs

Open an issue with:

- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python/Node version)

## 💡 Feature Requests

Open an issue tagged `enhancement` with a description of the proposed feature and its use case.
