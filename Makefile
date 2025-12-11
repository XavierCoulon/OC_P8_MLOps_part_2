# Development commands
.PHONY: up down rebuild precommit test coverage ui

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
