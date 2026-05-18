.DEFAULT_GOAL := help

# Use uv from PATH if available; fall back to the default install location
# so the same make invocation can continue right after auto-installing uv.
UV         := $(shell which uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
LOCAL_MODEL ?= qwen2.5:1.5b

.PHONY: help install pull-model demo serve test test-unit lint format clean check-ollama

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install uv (if needed) and all project dependencies
	@if ! which uv > /dev/null 2>&1; then \
		echo ""; \
		echo "  \033[33muv\033[0m is not installed — it manages this project's virtual environment and dependencies."; \
		printf "  Install it now? [y/N] "; \
		read answer; \
		case "$$answer" in \
			[yY]*) \
				echo "  \033[33m→ Installing uv...\033[0m"; \
				curl -LsSf https://astral.sh/uv/install.sh | sh; \
				echo "  \033[32m✓ uv installed\033[0m"; \
				echo ""; \
				;; \
			*) \
				echo ""; \
				echo "  Install it manually: https://docs.astral.sh/uv/getting-started/installation/"; \
				exit 1; \
				;; \
		esac; \
	fi
	$(UV) sync --all-extras

pull-model: check-ollama ## Download the local model via Ollama (~1 GB, one-time)
	@if ! ollama list > /dev/null 2>&1; then \
		echo "  \033[33m→ Starting ollama server...\033[0m"; \
		ollama serve > /dev/null 2>&1 & \
		until ollama list > /dev/null 2>&1; do sleep 1; done; \
		echo "  \033[32m✓ Server ready\033[0m"; \
	fi
	ollama pull $(LOCAL_MODEL)

demo: ## Run an interactive CLI demo of the multi-agent system
	$(UV) run python -m fitness_coach.demo

serve: ## Start the web UI (http://localhost:8000)
	$(UV) run uvicorn fitness_coach.web:app --reload --port 8000

test: ## Run the full test suite (requires Ollama or OPENAI_API_KEY)
	$(UV) run pytest -v

test-unit: ## Run only data-layer unit tests — no LLM required
	$(UV) run pytest -v -m "not integration"

lint: ## Check code style and imports
	$(UV) run ruff check .

format: ## Auto-fix formatting and import order
	$(UV) run ruff check --fix . && $(UV) run ruff format .

clean: ## Remove the virtual environment and all cache files
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

check-ollama:
	@if ! which ollama > /dev/null 2>&1; then \
		echo ""; \
		echo "  \033[33mollama\033[0m is not installed — it runs the local AI model for this project."; \
		printf "  Install it now? [y/N] "; \
		read answer; \
		case "$$answer" in \
			[yY]*) \
				echo "  \033[33m→ Installing ollama...\033[0m"; \
				if [ "$$(uname)" = "Darwin" ]; then \
					if which brew > /dev/null 2>&1; then \
						brew install ollama; \
					else \
						echo "  \033[31mHomebrew not found.\033[0m Install Ollama from: https://ollama.com/download/mac"; \
						exit 1; \
					fi; \
				else \
					curl -fsSL https://ollama.com/install.sh | sh; \
				fi; \
				echo "  \033[32m✓ ollama installed\033[0m"; \
				echo ""; \
				;; \
			*) \
				echo ""; \
				echo "  Install it manually: https://ollama.com/download"; \
				exit 1; \
				;; \
		esac; \
	fi
