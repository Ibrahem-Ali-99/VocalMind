.PHONY: help up down build logs backend-dev backend-test backend-lint backend-install frontend-dev frontend-build frontend-lint frontend-test frontend-install seed migrate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

build: ## Build all Docker images
	docker compose build

logs: ## Follow logs for all services
	docker compose logs -f

# ── Backend ───────────────────────────────────────────────────────────────

backend-dev: ## Run backend in dev mode
	cd backend && uv run uvicorn app.main:app --reload --port 8000

backend-test: ## Run backend tests
	cd backend && uv run pytest tests/ -v

backend-test-cov: ## Run backend tests with coverage
	cd backend && uv run pytest --cov=app --cov-report=term --cov-report=html tests/ -v

backend-lint: ## Lint backend code
	cd backend && uv run ruff check .

backend-install: ## Install backend dependencies
	cd backend && uv sync

# ── Frontend ──────────────────────────────────────────────────────────────

frontend-dev: ## Run frontend in dev mode
	cd frontend && npm run dev

frontend-build: ## Build frontend
	cd frontend && npm run build

frontend-lint: ## Lint frontend code
	cd frontend && npm run lint

frontend-test: ## Run frontend E2E tests (Cypress)
	cd frontend && npm run cy:run

frontend-test-cov: ## Run frontend tests with coverage
	cd frontend && npx vitest run --coverage.enabled --coverage.reporter=text --coverage.reporter=html

frontend-install: ## Install frontend dependencies
	cd frontend && npm ci

# ── Database ──────────────────────────────────────────────────────────────

seed: ## Seed the database
	cd backend && uv run python ../infra/scripts/seed_database.py

migrate: ## Run database migrations
	cd backend && uv run python ../infra/scripts/migrate.py

# ── Utilities ─────────────────────────────────────────────────────────────

clean: ## Remove all caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist
