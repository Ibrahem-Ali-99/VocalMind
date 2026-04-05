.PHONY: help up down build build-retry logs support-up support-down be-dev be-test be-test-cov be-lint be-install fe-dev fe-build fe-lint fe-test fe-e2e-summary fe-e2e-cov fe-test-cov fe-install rag-lint rag-test rag-install llm-trigger-test seed migrate clean test-all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

support-up: ## Start only supporting services for local backend/frontend development
	docker compose up -d db ollama qdrant vad emotion whisperx

support-down: ## Stop supporting services used by local backend/frontend development
	docker compose stop db ollama qdrant vad emotion whisperx

build: ## Build all Docker images
	docker compose build

build-retry: ## Build/start with retry logic for transient Docker daemon EOF/500 errors (Windows PowerShell)
	powershell -ExecutionPolicy Bypass -File infra/scripts/docker_compose_retry.ps1 -ComposeArgs "up -d --build"

logs: ## Follow logs for all services
	docker compose logs -f

# ── Backend ───────────────────────────────────────────────────────────────

be-dev: ## Run backend in dev mode
	cd backend && uv run uvicorn app.main:app --reload --port 8000

be-test: ## Run backend tests
	cd backend && uv run pytest tests/ -v

be-test-cov: ## Run backend tests with coverage
	cd backend && uv run pytest --cov=app --cov-report=term --cov-report=html tests/ -v

be-lint: ## Lint backend code
	cd backend && uv run ruff check .

be-install: ## Install backend dependencies
	cd backend && uv sync

# ── Frontend ──────────────────────────────────────────────────────────────

fe-dev: ## Run frontend in dev mode
	cd frontend && npm run dev

fe-build: ## Build frontend
	cd frontend && npm run build

fe-lint: ## Lint frontend code
	cd frontend && npm run lint

fe-test: ## Run frontend E2E tests (Cypress)
	cd frontend && npm run build
	cd frontend && npx -y concurrently -k -s first "npm run preview" "npx -y wait-on http-get://localhost:3000/ && npm run cy:run"

fe-e2e-summary: ## Run frontend E2E tests with concise summary
	cd frontend && npm run build
	cd frontend && npx -y concurrently -k -s first "npm run preview" "npx -y wait-on http-get://localhost:3000/ && npx cypress run --reporter list"

fe-e2e-cov: ## Run frontend E2E tests and generate code coverage report
	powershell -NoProfile -Command "$$paths = 'frontend/.nyc_output','frontend/coverage','frontend/dist'; foreach ($$path in $$paths) { if (Test-Path $$path) { Remove-Item -Recurse -Force $$path } }"
	powershell -NoProfile -Command "Set-Location frontend; $$env:CYPRESS_COVERAGE='true'; npm run build"
	cd frontend && npx -y concurrently -k -s first "npm run preview" "npx -y wait-on http-get://localhost:3000/ && npx cypress run --env coverage=true" && npx nyc report --reporter=html --reporter=text-summary

fe-test-cov: ## Run frontend unit tests with coverage report
	cd frontend && npx vitest run --coverage.enabled --coverage.reporter=text --coverage.reporter=html

fe-install: ## Install frontend dependencies
	cd frontend && npm ci

# ── RAG ───────────────────────────────────────────────────────────────────

rag-lint: ## Lint RAG code
	cd services/rag && uv run ruff check .

rag-test: ## Run RAG tests
	cd services/rag && uv run pytest tests/ -v

rag-install: ## Install RAG dependencies
	cd services/rag && uv sync

llm-trigger-test: ## Run full LLM-trigger validation (backend + RAG + frontend)
	cd backend && uv run pytest tests/test_llm_trigger_service.py tests/test_interactions_llm_triggers.py tests/test_sop_retrieval.py -q
	cd services/rag && uv run pytest tests/test_ingest.py -q
	cd frontend && npm run test -- --run src/tests/AgentCallDetail.test.tsx

# ── CI/CD ─────────────────────────────────────────────────────────────────

test-all: ## Run all tests required for CI/CD and clean up
	$(MAKE) -j test-backend test-frontend test-rag
	$(MAKE) clean

test-backend: be-lint be-test

test-frontend:
	$(MAKE) fe-lint
	$(MAKE) fe-build
	$(MAKE) fe-test-cov
	$(MAKE) fe-test

test-rag: rag-lint rag-test
# ── Database ──────────────────────────────────────────────────────────────

seed: ## Seed the database
	cd backend && uv run python ../infra/scripts/seed_database.py

migrate: ## Run database migrations
	cd backend && uv run python ../infra/scripts/migrate.py

# ── Utilities ─────────────────────────────────────────────────────────────

clean: ## Remove all caches and build artifacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .nyc_output -o -name coverage \) -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/.nyc_output
