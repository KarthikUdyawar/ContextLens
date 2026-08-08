SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

.PHONY: help up down down-v restart rebuild logs logs-api ps \
	shell-api shell-postgres health \
	pre-commit-install pre-commit lint fmt \
	test test-unit tree

.DEFAULT_GOAL := help

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

PG_USER := contextlens
PG_DB   := contextlens

COMPOSE := docker compose

GREEN := \033[0;32m
NC    := \033[0m

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────

help: ## Show available targets
	@echo ""
	@echo "ContextLens — available targets"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ { \
		printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────────────────────────────────────

up: ## Build + start all services detached
	$(COMPOSE) up --build -d
	echo -e "$(GREEN)✅ Stack is up$(NC)"
	echo "   API:      http://localhost:8000"
	echo "   API docs: http://localhost:8000/docs"
	echo "   MinIO:    http://localhost:9001"

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and delete volumes
	$(COMPOSE) down -v

restart: ## Restart stack
	$(MAKE) down
	$(MAKE) up

rebuild: ## Rebuild API image without cache
	$(COMPOSE) build --no-cache api

logs: ## Tail all logs
	$(COMPOSE) logs -f

logs-api: ## Tail API logs
	$(COMPOSE) logs -f api

ps: ## Show container status
	$(COMPOSE) ps

# ──────────────────────────────────────────────────────────────────────────────
# Shell access
# ──────────────────────────────────────────────────────────────────────────────

shell-api: ## Open bash in API container
	$(COMPOSE) exec api bash

shell-postgres: ## Open psql shell
	$(COMPOSE) exec postgres psql -U $(PG_USER) $(PG_DB)

# ──────────────────────────────────────────────────────────────────────────────
# Dev helpers
# ──────────────────────────────────────────────────────────────────────────────

health: ## Hit /health endpoint
	curl -s http://localhost:8000/health | python3 -m json.tool

# ──────────────────────────────────────────────────────────────────────────────
# Pre-commit / lint
# ──────────────────────────────────────────────────────────────────────────────

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

pre-commit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files --show-diff-on-failure

lint: ## Run lint + type checks
	uv run ruff check src tests
	uv run mypy src

fmt: ## Auto-format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

test: ## Run full test suite
	uv run pytest -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit -v

tree: ## Show project tree (respects .gitignore)
	tree --gitignore -I '__pycache__|*.pyc|*.egg-info' > docs/PROJECT.tree && echo 'Done'
