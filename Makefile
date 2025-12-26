# Development commands
.PHONY: up down rebuild precommit test coverage ui batch evaluate run-local

# Docker commands
up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose down -v
	docker compose build
	docker compose up -d

# Precommit commands
precommit:
	pre-commit run --all-files

# Run tests
test:
	pytest -v

# covergage report
coverage:
	pytest --cov=app --cov-report=term-missing --cov-report=html

# Launch Gradio UI
ui:
	python gradio_app.py

# Send batch predictions to API
batch:
	python scripts/batch_prediction.py

# Evaluate data drift
evaluate:
	python scripts/evaluate_drift.py

# Run app locally with profiling (uses .env.local)
run-local:
	@export $$(cat .env.local | grep -v '^#' | xargs) && \
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
