# XTAL — Makefile
# Convenient shortcuts for all common operations.
#
# Usage:
#   make up            — start the full Docker stack
#   make down          — stop the stack
#   make test          — run all unit + integration tests
#   make baseline      — run Week 2 baseline characterization
#   make controller    — start the XTAL controller
#   make eval          — run full evaluation (Week 5)
#   make ablation      — run ablation study
#   make figures       — generate all paper figures from latest results
#   make paper         — compile LaTeX paper (requires pdflatex)
#   make clean         — remove generated outputs

PYTHON   := python3
SLA      := 500
RATE     := 5000
DURATION := 120
RUNS     := 3

# ── Infrastructure ────────────────────────────────────────────────────────────

.PHONY: up down restart logs status clean-volumes

up:
	docker compose up -d
	@echo "Flink UI:   http://localhost:8081"
	@echo "Grafana:    http://localhost:3000  (admin / xtal2024)"
	@echo "Prometheus: http://localhost:9090"

down:
	docker compose down

restart:
	docker compose down && docker compose up -d

logs:
	docker compose logs -f

status:
	docker compose ps

clean-volumes:
	docker compose down -v

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install setup

install:
	pip install -r requirements.txt

setup: install up
	@sleep 10
	bash scripts/setup.sh

# ── Jobs ─────────────────────────────────────────────────────────────────────

.PHONY: submit-job load

submit-job:
	bash jobs/nexmark_submit.sh 5 4 600

load:
	$(PYTHON) scripts/load_generator.py --rate $(RATE) \
	  --burst-at 30 --burst-rate $(shell echo $$(($(RATE)*10))) --duration 120

# ── Tests (Week 4) ────────────────────────────────────────────────────────────

.PHONY: test test-unit test-integration coverage

test: test-unit test-integration

test-unit:
	pytest tests/test_reward.py tests/test_agent.py tests/test_actuator.py tests/test_telemetry.py \
	  -v --tb=short

test-integration:
	pytest tests/integration_test.py -v --tb=short

coverage:
	pytest tests/ --cov=controller --cov-report=term-missing --cov-report=html:htmlcov

# ── Controller (Week 3) ───────────────────────────────────────────────────────

.PHONY: controller

controller:
	$(PYTHON) -m controller.main \
	  --sla $(SLA) \
	  --interval 10 \
	  --model-path models/xtal_agent.pkl

# ── Baseline (Week 2) ─────────────────────────────────────────────────────────

.PHONY: baseline baseline-quick

baseline:
	$(PYTHON) -m experiments.baseline_runner \
	  --duration $(DURATION) \
	  --rates $(RATE) \
	  --sla $(SLA)

baseline-quick:
	$(PYTHON) -m experiments.baseline_runner \
	  --duration 60 \
	  --rates $(RATE) \
	  --configs flink-default hand-tuned \
	  --patterns steady bursty

# ── Full Evaluation (Week 5) ──────────────────────────────────────────────────

.PHONY: eval eval-quick ablation figures

eval:
	$(PYTHON) -m experiments.evaluate \
	  --duration 300 \
	  --runs $(RUNS) \
	  --base-rate $(RATE) \
	  --sla $(SLA)

eval-quick:
	$(PYTHON) -m experiments.evaluate \
	  --duration 120 \
	  --runs 1 \
	  --patterns bursty

ablation:
	$(PYTHON) -m experiments.ablation \
	  --duration 180 \
	  --runs 2 \
	  --patterns bursty failure

figures:
	@LATEST_RAW=$$(ls -t experiments/results/eval_raw_*.csv 2>/dev/null | head -1); \
	LATEST_DEC=$$(ls -t logs/xtal_decisions_*.csv 2>/dev/null | head -1); \
	if [ -z "$$LATEST_RAW" ]; then \
	  echo "No eval results found. Run 'make eval' first."; exit 1; \
	fi; \
	CMD="$(PYTHON) -m experiments.plot_results --raw $$LATEST_RAW"; \
	if [ -n "$$LATEST_DEC" ]; then CMD="$$CMD --decisions $$LATEST_DEC"; fi; \
	echo "Running: $$CMD"; \
	$$CMD

# ── Failure injection (standalone) ───────────────────────────────────────────

.PHONY: fail-node fail-network fail-cpu fail-checkpoint

fail-node:
	$(PYTHON) -m experiments.failure_injector node-failure --duration 30

fail-network:
	$(PYTHON) -m experiments.failure_injector network-delay --delay-ms 200 --duration 60

fail-cpu:
	$(PYTHON) -m experiments.failure_injector cpu-stress --workers 2 --duration 30

fail-checkpoint:
	$(PYTHON) -m experiments.failure_injector checkpoint-hang --pause 15

# ── Paper (Week 6) ────────────────────────────────────────────────────────────

.PHONY: paper

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex \
	         && bibtex main \
	         && pdflatex -interaction=nonstopmode main.tex \
	         && pdflatex -interaction=nonstopmode main.tex
	@echo "Paper compiled: paper/main.pdf"

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean

clean:
	rm -rf experiments/results/*.csv experiments/results/*.json
	rm -rf experiments/figures/*.pdf
	rm -rf logs/*.csv
	rm -rf models/*.pkl
	rm -rf htmlcov/ .coverage
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "Cleaned all generated outputs."

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "XTAL — eXplainable Tail-latency Adaptive controLler"
	@echo ""
	@echo "  make setup          Install deps + start Docker stack"
	@echo "  make test           Run all tests"
	@echo "  make coverage       Tests with HTML coverage report"
	@echo "  make submit-job     Submit NEXMark Q5 to Flink"
	@echo "  make controller     Start XTAL control loop"
	@echo "  make baseline       Week 2: baseline characterization"
	@echo "  make eval           Week 5: full evaluation (3 baselines × 3 runs)"
	@echo "  make ablation       Week 5: ablation study"
	@echo "  make figures        Generate all paper figures"
	@echo "  make paper          Compile LaTeX paper"
	@echo "  make clean          Remove all generated outputs"
	@echo ""
