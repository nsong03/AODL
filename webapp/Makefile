# AODL Studio — developer entry points (W1 work order §6; target names are frozen).
#
#   make install        editable installs (root aodl + backend) and `npm ci`
#   make dev            backend on :8000 (--reload) and the Vite dev server on :5173
#   make test           backend + frontend test suites (strict advanced-router import)
#   make lint           ruff + mypy on the backend, eslint + tsc on the frontend
#   make build          production frontend bundle -> frontend/dist
#   make fixtures       regenerate the golden JSON fixtures the frontend tests use
#   make serve          one uvicorn on :8000 serving the API and the built frontend

SHELL := /bin/bash
PYTHON ?= python
NPM ?= npm
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BACKEND := $(ROOT)/webapp/backend
FRONTEND := $(ROOT)/webapp/frontend

# A broken or missing advanced router must fail the tests, never ship silently (DESIGN §3).
export AODL_STUDIO_STRICT_ADVANCED ?= 1

.PHONY: install dev dev-backend dev-frontend test test-backend test-frontend lint lint-backend \
        lint-frontend build fixtures serve

install:
	$(PYTHON) -m pip install -e "$(ROOT)[dev]" -e "$(BACKEND)[dev]"
	cd $(FRONTEND) && $(NPM) ci

dev-backend:
	cd $(BACKEND) && AODL_STUDIO_STRICT_ADVANCED=0 $(PYTHON) -m uvicorn aodl_webapp.main:create_app \
		--factory --reload --port $(BACKEND_PORT)

dev-frontend:
	cd $(FRONTEND) && $(NPM) run dev -- --port $(FRONTEND_PORT)

# Both servers in one terminal; Ctrl-C stops both.
dev:
	@echo "AODL Studio dev: API http://localhost:$(BACKEND_PORT)/api  UI http://localhost:$(FRONTEND_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	( cd $(BACKEND) && AODL_STUDIO_STRICT_ADVANCED=0 $(PYTHON) -m uvicorn aodl_webapp.main:create_app \
		--factory --reload --port $(BACKEND_PORT) ) & \
	( cd $(FRONTEND) && $(NPM) run dev -- --port $(FRONTEND_PORT) ) & \
	wait

test: test-backend test-frontend

test-backend:
	cd $(BACKEND) && $(PYTHON) -m pytest tests -q

test-frontend:
	cd $(FRONTEND) && $(NPM) test -- --run

lint: lint-backend lint-frontend

lint-backend:
	cd $(ROOT) && ruff check webapp/backend webapp/scripts
	cd $(ROOT) && ruff format --check webapp/backend webapp/scripts
	cd $(BACKEND) && $(PYTHON) -m mypy src/aodl_webapp src/aodl_advanced

lint-frontend:
	cd $(FRONTEND) && $(NPM) run lint
	cd $(FRONTEND) && $(NPM) run typecheck

build:
	cd $(FRONTEND) && $(NPM) run build

fixtures:
	$(PYTHON) $(ROOT)/webapp/scripts/gen_frontend_fixtures.py

serve: build
	@echo "AODL Studio: http://localhost:$(BACKEND_PORT) (API under /api; frontend from frontend/dist)"
	cd $(BACKEND) && $(PYTHON) -m uvicorn aodl_webapp.main:create_app --factory --port $(BACKEND_PORT)
