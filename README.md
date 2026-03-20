# XTAL — eXplainable Tail-latency Adaptive controLler

A real implementation of a closed-loop, cross-layer tail-latency controller for strict-deadline distributed streaming pipelines.

**Authors:** Akanksha Garg, Hrithik Gaikwad — Rochester Institute of Technology

---

## What is XTAL?

Modern distributed streaming pipelines (e.g., real-time analytics, autonomous vehicle perception) must meet strict latency deadlines. The problem is that **tail latency** (p99/p999) — not average latency — determines deadline misses. Existing systems like Flink and Kafka have no built-in closed-loop controller that actively reduces tail latency.

XTAL fills this gap: it monitors real-time telemetry from a Kafka + Flink pipeline and uses an **XGBoost contextual bandit with SHAP explanations** to tune system knobs (checkpoint intervals, buffer timeouts, Kafka linger) in a continuous closed loop to keep p99/p999 latency within SLA bounds.

**Novel contributions for the paper:**
1. First closed-loop, cross-layer tail-latency controller for a real Kafka + Flink stack
2. SHAP explanations per control action — operators see *why* each knob was changed
3. Safety constraint layer — hard guardrails that prevent correctness violations during tuning
4. Evaluation against 3 baselines on NEXMark benchmarks under burst load and failure injection

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streaming Pipeline                │
│  Kafka (ingest) → Flink (operators) → Sink          │
│       ↑ backpressure    ↑ checkpoints               │
└────────────────────┬────────────────────────────────┘
                     │ telemetry (Prometheus/JMX)
┌────────────────────▼────────────────────────────────┐
│              XTAL Control Plane (Python)            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Telemetry  │  │  XGBoost     │  │   SHAP    │  │
│  │  Collector  │→ │  Bandit      │→ │  (XAI)    │  │
│  └─────────────┘  └──────┬───────┘  └───────────┘  │
│                          │ actions                  │
│  ┌───────────────────────▼─────────────────────┐   │
│  │  Actuator API (Flink REST + Kafka AdminAPI)  │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Knobs XTAL tunes in real-time:**

| Layer | Knob | Effect |
|-------|------|--------|
| Flink | `checkpoint_interval_ms` | Reduces checkpoint overhead / state sync lag |
| Flink | `buffer_timeout_ms` | Controls output buffer flush latency (batching trade-off) |
| Kafka | `linger.ms` | Producer batching delay |

**Control loop timing:** decisions every 10 seconds (configurable). SHAP explanation generated for every action. Decision log written to `logs/xtal_decisions_<timestamp>.csv`.

---

## Repository Structure

```
xtal/
├── docker-compose.yml               # Full 8-container stack
├── requirements.txt                 # Python dependencies
│
├── config/
│   ├── flink/
│   │   └── flink-conf.yaml          # Prometheus metrics, checkpointing, RocksDB
│   ├── prometheus/
│   │   └── prometheus.yml           # Scrape targets: JM + 3 TMs + Kafka
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/         # Auto-connects Prometheus
│       │   └── dashboards/          # Auto-loads dashboard
│       └── dashboards/
│           └── xtal.json            # Pre-built dashboard: p99/p999, buffer, lag
│
├── controller/                      # ── WEEK 1 & 3 ──
│   ├── telemetry.py                 # Prometheus queries → SystemState dataclass
│   ├── actuator.py                  # Flink REST + Kafka AdminClient wrappers
│   ├── agent.py                     # XGBoost contextual bandit + SHAP explanations
│   ├── reward.py                    # Reward function (p99, SLA penalty, throughput)
│   └── main.py                      # Closed-loop controller entry point
│
├── experiments/                     # ── WEEK 2 & 5 ──
│   ├── baseline_runner.py           # Runs 4 configs × 3 load patterns, saves CSV
│   ├── failure_injector.py          # 4 failure modes: node/network/cpu/checkpoint
│   ├── evaluate.py                  # Full evaluation: XTAL vs 3 baselines
│   ├── ablation.py                  # Ablation: no-safety, single-layer variants
│   └── plot_results.py              # Generates 6 paper-quality PDF figures
│
├── tests/                           # ── WEEK 4 ──
│   ├── test_reward.py               # 9 unit tests for reward function
│   ├── test_agent.py                # 9 unit tests for XGBoost agent
│   ├── test_actuator.py             # 12 unit tests for safety bounds
│   ├── test_telemetry.py            # 7 unit tests for telemetry collector
│   └── integration_test.py          # End-to-end loop tests (mocked infrastructure)
│
├── paper/                           # ── WEEK 6 ──
│   ├── main.tex                     # ACM sigconf LaTeX entry point
│   ├── xtal.bib                     # Bibliography (25 references)
│   └── sections/
│       ├── 01_intro.tex             # Introduction + contributions
│       ├── 02_background.tex        # Background: Flink, Kafka, XAI, bandits
│       ├── 03_motivation.tex        # Baseline characterization findings
│       ├── 04_design.tex            # XTAL design: state/action/reward/SHAP/safety
│       ├── 05_implementation.tex    # Implementation details + overhead numbers
│       ├── 06_evaluation.tex        # Results tables + figure references
│       ├── 07_related.tex           # Related work
│       └── 08_conclusion.tex        # Conclusion + future work
│
├── Makefile                         # Shortcuts for all common operations
├── jobs/
│   └── nexmark_submit.sh            # Downloads and submits NEXMark benchmark jar
│
└── scripts/
    ├── setup.sh                     # One-command bootstrap
    └── load_generator.py            # NEXMark events with burst injection
```

---

## Code Walkthrough

### The Big Picture

XTAL is a **feedback control system**. Every 10 seconds it runs this loop:

```
observe system → pick a config → apply it → measure what happened → learn
```

The "system" is a Kafka + Flink streaming pipeline. The "config" is a combination of three knobs. The "learning" is an ML model that gets progressively better at predicting which config will reduce tail latency.

### Data Flow

```
load_generator.py
      │ events (JSON)
      ▼
    Kafka ──► Flink ──► (results)
      │          │
      │          │ metrics (Prometheus format)
      ▼          ▼
  kafka-exporter  Flink Prometheus reporter
      │                │
      └────────────────┘
                │
           Prometheus
                │
      TelemetryCollector.collect()
                │ SystemState (6 numbers)
                ▼
         XTALAgent.select_action()
                │ (action_idx, SHAP explanation)
                ▼
         XTALActuator.apply_action()
                │ Flink REST API / Kafka AdminClient
                ▼
      Flink checkpoint interval updated
      Kafka linger hint file written
                │
         (next iteration, 10s later)
```

---

### Layer 1 — Infrastructure (`docker-compose.yml`)

Eight containers started with `docker compose up`:

```
Kafka ──────────────────────────────────► Flink (processes data)
  │                                          │
  └── kafka-exporter ──► Prometheus ◄─────── ┘  (both export metrics)
                              │
                           Grafana  (visualises metrics)
```

**Kafka** receives events (auction bids, in NEXMark format). **Flink** reads from Kafka, runs streaming operators (joins, windows, aggregations), and writes results. Both export metrics to **Prometheus** every 5 seconds. **Grafana** queries Prometheus and renders the pre-built XTAL dashboard.

`config/flink/flink-conf.yaml` is critical — it tells Flink to expose its metrics in Prometheus format on port 9249, enables latency tracking per operator, and configures RocksDB checkpointing.

---

### Layer 2 — Telemetry (`controller/telemetry.py`)

How XTAL "sees" the system. Sends 6 PromQL queries to Prometheus every 10 seconds and packages the results into a `SystemState`:

```python
@dataclass
class SystemState:
    p99_latency_ms       # end-to-end latency at 99th percentile
    p999_latency_ms      # end-to-end latency at 99.9th percentile
    input_buffer_usage   # how full Flink's network buffers are (0.0–1.0)
    checkpoint_duration_ms  # how long the last checkpoint took
    kafka_consumer_lag   # unread messages sitting in Kafka
    records_in_per_sec   # current throughput
```

If Prometheus is unreachable or a metric returns nothing, every field defaults to `0.0` — the controller keeps running safely. `as_feature_vector()` strips the timestamp and returns a plain list of 6 floats — that's the input to the ML model.

---

### Layer 3 — Actuator (`controller/actuator.py`)

How XTAL "acts" on the system. Three classes:

**`ControlAction`** — a data class holding what to change (`None` means don't touch that knob):
```python
ControlAction(checkpoint_interval_ms=2000, buffer_timeout_ms=10, kafka_linger_ms=5)
```
`is_safe()` enforces hard bounds before anything is sent to the cluster:
- Checkpoint: 500ms – 30,000ms
- Parallelism: 1 – 12
- Buffer timeout: 0 – 200ms

**`FlinkActuator`** — wraps the Flink REST API. Key methods: `get_running_jobs()`, `trigger_checkpoint()`, `rescale_job()` (stop-with-savepoint → restart at new parallelism — disruptive, used sparingly), `set_job_config_dynamic()` (runtime config update without restart, supported for a limited set of keys in Flink 1.17+).

**`KafkaActuator`** — wraps the Kafka AdminClient. Kafka's `linger.ms` is a producer config, not a topic config, so it's communicated via a shared hint file (`/tmp/xtal_producer_hints.json`) that the load generator reads and applies to its producer.

**`XTALActuator`** — the one `main.py` uses. Calls `is_safe()` before touching anything:
```python
def apply_action(self, action, job_id, topic):
    if not action.is_safe():
        log.error("Unsafe action rejected")  # config stays unchanged
        return False
    # ... call Flink REST, write Kafka hints
```

---

### Layer 4 — Reward (`controller/reward.py`)

Tells the ML model whether the last decision was good or bad. Returns a single float:

```
reward = -(p99 / 500)          # always negative; 0 = perfect, -1 = at SLA boundary
       + (-20 if p99 > 500)    # hard cliff: SLA violation is heavily penalised
       + throughput_term        # small bonus/penalty for throughput change
       + (-1 if ckpt > 10s)    # checkpoint storm = bad (operators are stalling)
       + (-2 if buf_usage > 0.9) # severe backpressure = bad (latency spike imminent)
```

Why this design:
- The **latency term** is proportional — sitting at 490ms is almost as bad as 510ms
- The **SLA cliff** (-20) makes the model strongly avoid ever crossing 500ms
- **Throughput** is secondary — we don't want the controller to sacrifice throughput just to shave latency
- **Checkpoint storm** catches the case where a very long checkpoint is blocking operator progress
- **Backpressure** fires when Flink's buffers are nearly full — this is a *leading indicator* that latency is about to spike, so penalising it teaches the agent to act proactively

---

### Layer 5 — Agent (`controller/agent.py`)

The brain. Uses a **contextual bandit** — a simplified form of reinforcement learning where there is no credit-assignment across time; just "given this state, what action gives the best immediate reward?"

**Action space — 14 fixed configurations:**

```
Action 0:  ckpt=500ms,   buf=0ms,   linger=0ms   ← most aggressive
Action 6:  ckpt=2000ms,  buf=10ms,  linger=5ms   ← balanced
Action 13: ckpt=10000ms, buf=100ms, linger=50ms  ← most conservative
```

All 14 are pre-validated safe. The agent never generates out-of-bounds actions.

**The model:** One XGBoost regressor. Input = 6 state features + 3 action features (9 total). Output = predicted reward for taking that action in the current state. To choose an action, the model is evaluated for all 14 candidates and the one with the highest predicted reward is selected.

**Exploration vs exploitation (epsilon-greedy):**
```
steps 0–500:  ε decays 0.30 → 0.05
any step:     with prob ε  → random action  (explore the space)
              with prob 1-ε → best predicted action  (exploit what was learned)
```
Early on, the model knows nothing, so you explore randomly. Over time it exploits more.

**Safety override:** If `p99 > 400ms` (80% of SLA), the model subtracts 5.0 from the predicted reward of all conservative actions (checkpoint > 2000ms) before argmax. This biases the agent toward aggressive configs when pressure is building — it doesn't wait for a violation to react.

**Online learning:** Every 20 steps the model is retrained on all experience collected so far. This is "full replay" — using the entire history, not just the most recent transitions, which is more stable for a small dataset.

**SHAP explanations:** After each exploitative decision, `shap.TreeExplainer` computes how much each of the 9 input features pushed the predicted reward up or down for the chosen action. The top 3 are logged. Example:
```
input_buffer_usage: ↓ decreases reward by 0.41  ← buffer is high (hurting us)
p99_latency_ms:     ↑ increases reward by 0.28  ← latency is still ok
checkpoint_ms:      ↑ increases reward by 0.10  ← reducing ckpt helps
```
Translation: "buffer is high but latency hasn't spiked yet — act now by reducing checkpoint overhead."

---

### Layer 6 — Control Loop (`controller/main.py`)

Ties everything together. `XTALController.run()` executes this loop forever:

```
Step N:
  1. collect()           →  SystemState (6 metrics from Prometheus)
  2. compute_reward(     →  float (how did the previous action do?)
       prev_state,
       curr_state)
  3. agent.record(...)   →  store transition, retrain every 20 steps
  4. select_action()     →  (action_idx, SHAP explanation)
  5. apply_action()      →  update Flink REST + write Kafka hint
  6. log to CSV          →  state + action + reward + SHAP written to file
  7. sleep(10s)          →  wait for next interval
```

Note the order: reward is computed *before* selecting the next action, because reward measures the outcome of the *previous* action. On the very first step there is no previous action so reward is skipped.

Every 50 steps, the agent is saved to `models/xtal_agent.pkl`. On Ctrl-C or SIGTERM, it saves and closes the log cleanly.

---

### Layer 7 — Baseline Experiments (`experiments/`)

**`baseline_runner.py`** — Runs the uncontrolled pipeline under 4 static configs × 3 load patterns. No ML, no controller. Purpose: document *where tail spikes come from and how severe they are* without XTAL. This data is paper §3 (Motivation). It writes a CSV with p50/p99/p999/buffer/checkpoint per 5-second window.

**`failure_injector.py`** — Four failure modes you can trigger on-demand or inject mid-experiment:
- `node-failure` — `docker stop` a TaskManager, Flink detects it and restarts the job
- `network-delay` — `tc netem` adds artificial delay on a container's network interface (noisy network simulation)
- `cpu-stress` — `stress-ng` burns CPU inside a container (noisy neighbour CPU contention)
- `checkpoint-hang` — `docker pause` sends SIGSTOP, freezing the JVM; simulates a GC pause that blocks checkpoint barrier alignment

**`evaluate.py`** — Full comparison: 4 systems (XTAL + 3 baselines) × 3 patterns × 3 runs. For the reactive scaler baseline, it applies a simple rule: if `p99 > SLA` switch to low-latency preset; if `p99 < 0.5×SLA` relax back to defaults. This is representative of rule-based production autoscalers.

**`ablation.py`** — Runs 3 variants of XTAL to isolate which components contribute:
- `xtal-no-safety` — removes the SLA-pressure bias; agent may pick a conservative action at the worst moment
- `xtal-single-layer` — forces Kafka linger to 0 regardless of the chosen action, so only Flink knobs change; shows that cross-layer tuning matters

**`plot_results.py`** — Reads CSVs and generates 8 publication-quality PDFs. Fonts are embedded (`pdf.fonttype=42`) to satisfy ACM/IEEE submission requirements. Figures include: CDF of p99 per pattern, time series with annotated burst/failure windows, SLA violation bar chart, SHAP importance bar chart, tail amplification boxplot.

---

### Layer 8 — Tests (`tests/`)

All 50 tests run without Docker, Kafka, or Flink — everything that touches the network is mocked with `unittest.mock.patch`.

| File | What it validates |
|------|------------------|
| `test_reward.py` | Does the SLA cliff fire at exactly 500ms? Does NaN ever occur with extreme inputs? Does backpressure reduce reward? |
| `test_actuator.py` | Every safety bound: checkpoint min/max, parallelism min/max, buffer min/max, mixed valid+invalid fields, None fields (no-op) |
| `test_telemetry.py` | Mocked Prometheus: happy path, empty result set, HTTP timeout, SLA check method |
| `test_agent.py` | Action space covers correct range, epsilon decays to floor, model trains after 14+ samples, save/load roundtrip preserves step count |
| `integration_test.py` | Full 30-step control loop produces valid actions, model trains after 20 steps, SHAP fires after training, unsafe action is rejected before reaching Flink, decision CSV has all expected columns |

---

### Layer 9 — Paper (`paper/`)

ACM `sigconf` format (same as SOSP, EuroSys, Middleware). Each section is a separate `.tex` file in `paper/sections/`, included by `paper/main.tex`. `paper/xtal.bib` has all 25 citations. The figures in `experiments/figures/` are referenced directly from `06_evaluation.tex`.

Compile with:
```bash
make paper   # pdflatex → bibtex → pdflatex × 2 → paper/main.pdf
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker Desktop | ≥ 4.x | Runs all infrastructure containers |
| Python | ≥ 3.10 | Control plane and scripts |
| pip | ≥ 23.x | Python package installer |

Verify:
```bash
docker --version
docker compose version
python3 --version
```

---

## Week 1 — Infrastructure Setup

### Step 1 — Enter the project directory

```bash
cd /Users/akankshagarg/xtal
```

### Step 2 — Bootstrap the full stack

One command installs Python deps, starts all containers, creates Kafka topics, and verifies every endpoint:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Expected output at the end:
```
✓ Flink REST API → http://localhost:8081/overview
✓ Prometheus     → http://localhost:9090/-/healthy
✓ Grafana        → http://localhost:3000/api/health
✓ Kafka Exporter → http://localhost:9308/metrics
```

If a service shows `✗ unreachable`, wait 30 seconds — containers may still be starting.

### Step 3 — Open the UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Flink Web UI | http://localhost:8081 | none |
| Prometheus | http://localhost:9090 | none |
| Grafana | http://localhost:3000 | admin / xtal2024 |

### Step 4 — Submit a NEXMark streaming job

```bash
./jobs/nexmark_submit.sh 5 4 300
# args: [query_number] [parallelism] [duration_seconds]
```

If the NEXMark jar download fails, use Flink's built-in example:
```bash
docker exec xtal-jobmanager flink run /opt/flink/examples/streaming/StateMachineExample.jar
```

### Step 5 — Verify telemetry is flowing

```bash
python3 -c "from controller.telemetry import TelemetryCollector; print(TelemetryCollector().collect())"
```

You should see a `SystemState` with non-zero values once a job is running.

### Step 6 — Generate bursty load and confirm the latency spike in Grafana

```bash
python3 scripts/load_generator.py --rate 5000 --burst-at 30 --burst-rate 50000 --duration 120
```

Open http://localhost:3000 → **XTAL — Tail Latency Control** dashboard. The p99 spike at t=30s is the problem XTAL solves.

---

## Week 2 — Baseline Characterization

This step characterizes the *uncontrolled* pipeline under different static configurations and load patterns. The results form the motivation section of the paper.

### Configurations tested

| Name | checkpoint_ms | buffer_timeout_ms | kafka_linger_ms | Notes |
|------|-------------|-----------------|----------------|-------|
| `flink-default` | 5000 | 100 | 5 | Flink 1.18 out-of-the-box |
| `low-latency` | 500 | 0 | 0 | Aggressively tuned for minimum latency |
| `high-throughput` | 10000 | 100 | 50 | Tuned for maximum throughput |
| `hand-tuned` | 2000 | 10 | 5 | Best static config from manual tuning |

### Load patterns

| Pattern | Description |
|---------|-------------|
| `steady` | Constant 5,000 events/s |
| `bursty` | 5,000/s with 10× spike (50,000/s) at t=30s for 10s |
| `failure` | Steady load + TaskManager killed at t=45s, restarted at t=75s |

### Run baseline experiments

```bash
# Full matrix: 4 configs × 3 patterns, 120s each (≈ 25 minutes total)
python3 -m experiments.baseline_runner --duration 120 --rates 5000

# Quick smoke test: just 2 configs, steady only, 60s each
python3 -m experiments.baseline_runner --duration 60 --configs flink-default hand-tuned --patterns steady bursty
```

Results are saved to `experiments/results/baseline_<timestamp>.csv`.

### Failure injection (standalone)

```bash
# Kill TaskManager for 30s then restart
python3 -m experiments.failure_injector node-failure --target xtal-taskmanager-3 --duration 30

# Add 200ms network delay on Kafka for 60s
python3 -m experiments.failure_injector network-delay --delay-ms 200 --duration 60

# Simulate checkpoint hang (SIGSTOP a TaskManager for 15s)
python3 -m experiments.failure_injector checkpoint-hang --target xtal-taskmanager-2 --pause 15

# CPU stress (noisy neighbour simulation)
python3 -m experiments.failure_injector cpu-stress --target xtal-taskmanager-1 --workers 2 --duration 30
```

### What to record from Week 2

For each configuration + pattern combination, note from the CSV:
- Peak p99 latency during the stress event
- Time to recover (p99 back below 500ms SLA)
- Which signal spiked first: `input_buffer_usage` vs `checkpoint_duration_ms`

This data is paper §3 (Motivation).

---

## Week 3 — XTAL Control Plane

### How the agent works

**State space** (6 features from Prometheus):

| Feature | Description |
|---------|-------------|
| `p99_latency_ms` | p99 end-to-end latency |
| `p999_latency_ms` | p999 end-to-end latency |
| `input_buffer_usage` | Fraction of input buffers full (backpressure proxy, 0–1) |
| `checkpoint_duration_ms` | Duration of last completed checkpoint |
| `kafka_consumer_lag` | Total consumer group lag across all partitions |
| `records_in_per_sec` | Ingestion throughput |

**Action space** (14 discrete preset configurations):

| # | checkpoint_ms | buffer_timeout_ms | kafka_linger_ms | Character |
|---|-------------|-----------------|----------------|-----------|
| 0 | 500 | 0 | 0 | Most aggressive |
| 1–4 | 500–1000 | 0–10 | 0–10 | Low latency |
| 5–7 | 2000 | 0–50 | 0–25 | Balanced |
| 8–10 | 5000 | 10–100 | 10–50 | Default range |
| 11–13 | 10000 | 10–100 | 10–50 | Most conservative |

**Reward function** (`controller/reward.py`):

```
reward = -(p99 / sla_ms)              # primary: latency term (0 to -∞)
       + SLA_PENALTY if p99 > SLA     # hard penalty: -20.0
       + throughput_term               # secondary: throughput stability
       + checkpoint_storm_penalty      # -1.0 if checkpoint > 10s
       + backpressure_penalty          # -2.0 if buffer_usage > 0.9
```

**Learning algorithm:** XGBoost contextual bandit
- Single XGBoost regressor: `(state_features + action_features) → predicted reward`
- Epsilon-greedy exploration: ε decays 0.30 → 0.05 over 500 steps
- Online retraining every 20 steps on the full experience buffer
- SHAP TreeExplainer generates top-3 feature attributions for every selected action

**Safety override:** if `p99 > 0.8 × SLA`, the agent penalises all actions with `checkpoint_ms > 2000` before argmax — preventing conservative choices when under pressure.

### Run the XTAL controller

```bash
# Auto-detect the running Flink job, 500ms SLA, 10s control interval
python3 -m controller.main --sla 500 --interval 10

# Explicit job ID, custom model path
python3 -m controller.main --sla 500 --interval 10 --job-id <flink-job-id> --model-path models/xtal.pkl

# All options
python3 -m controller.main --help
```

The controller logs every decision to `logs/xtal_decisions_<timestamp>.csv`, including:
- Full system state at decision time
- Action chosen and its configuration
- SHAP top-3 feature attributions and values
- Predicted and actual reward

### Sample decision log output

```
2026-03-19 10:00:10 INFO  [step 1] p99=312.4ms buf=0.12 ckpt=850ms lag=0 tps=5021
2026-03-19 10:00:10 INFO  Action #6: checkpoint=2000ms, buffer_timeout=10ms, kafka_linger=5ms
                            Predicted reward: -0.625
                            Key drivers:
                              input_buffer_usage: ↓ decreases reward by 0.412
                              p99_latency_ms:     ↑ increases reward by 0.284
                              checkpoint_ms:      ↑ increases reward by 0.103
```

### Save and reload agent

The agent is saved to `models/xtal_agent.pkl` every 50 steps and on shutdown. On restart, pass `--model-path` to resume from checkpoint:

```bash
python3 -m controller.main --model-path models/xtal_agent.pkl
```

---

## Week 5 — Full Evaluation

### Run the evaluation suite

Compares XTAL against 3 baselines under all load patterns:

```bash
# 3 repetitions, 300s each, all patterns (≈ 3 hours total)
python3 -m experiments.evaluate --duration 300 --runs 3

# Quick test: 1 run, 120s, bursty only
python3 -m experiments.evaluate --duration 120 --runs 1 --patterns bursty
```

**Baselines:**
1. `static-default` — Flink 1.18 out-of-the-box defaults
2. `static-hand-tuned` — Best static config from Week 2 characterization
3. `reactive-scaler` — Threshold-based: switches to low-latency preset when p99 > SLA, relaxes when p99 < 0.5×SLA

**Metrics collected per 5s window:**

| Metric | Description |
|--------|-------------|
| p50 / p99 / p999 | Latency percentiles (ms) |
| SLA violation rate | % of windows where p99 > 500ms |
| Tail amplification | p999 / p50 ratio |
| Throughput stability | Mean and std dev of records/s |
| Recovery time | Seconds from failure injection to p99 < SLA |

Results saved to:
- `experiments/results/eval_raw_<timestamp>.csv` — full time series
- `experiments/results/eval_summary_<timestamp>.json` — aggregate stats per system/pattern

### Generate paper figures

```bash
python3 -m experiments.plot_results \
  --raw experiments/results/eval_raw_<timestamp>.csv \
  --decisions logs/xtal_decisions_<timestamp>.csv
```

Figures saved to `experiments/figures/` (PDF, ACM/IEEE fonts embedded):

| File | Content |
|------|---------|
| `fig1_latency_cdf_bursty.pdf` | CDF of p99 latency — all systems, bursty load |
| `fig1_latency_cdf_failure.pdf` | CDF of p99 latency — all systems, failure load |
| `fig2_timeseries_burst.pdf` | p99 time series during burst event (annotated) |
| `fig3_timeseries_failure.pdf` | p99 time series during TaskManager failure |
| `fig4_sla_violation.pdf` | SLA violation rate bar chart per system × pattern |
| `fig5_shap_importance.pdf` | Mean SHAP feature importance across all XTAL decisions |
| `fig6_tail_amplification.pdf` | p999/p50 tail amplification boxplot |

---

## Week 4 — Testing & Integration

### Run all tests

```bash
make test          # unit tests + integration tests
make coverage      # same + HTML coverage report in htmlcov/
```

Or directly with pytest:
```bash
# Unit tests only (no live infrastructure needed)
pytest tests/test_reward.py tests/test_agent.py tests/test_actuator.py tests/test_telemetry.py -v

# Integration tests (mocked Prometheus + Flink — no live infrastructure needed)
pytest tests/integration_test.py -v
```

### What is tested

| File | Tests | Requires live stack? |
|------|-------|----------------------|
| `test_reward.py` | Reward function: SLA penalty, backpressure, checkpoint storm, NaN safety | No |
| `test_agent.py` | Action space bounds, epsilon decay, model trains after threshold, save/load | No |
| `test_actuator.py` | Safety bounds: checkpoint/parallelism/buffer min+max, mixed valid/invalid | No |
| `test_telemetry.py` | Prometheus mock: normal, empty result, timeout, SLA check | No |
| `integration_test.py` | Full 30-step control loop, SHAP after training, safety rejection, CSV columns | No |

---

## Week 5 — Ablation Study

```bash
# Ablation: XTAL-full vs no-safety vs single-layer (Flink only)
make ablation
# or
python3 -m experiments.ablation --duration 180 --runs 2 --patterns bursty failure
```

Results saved to `experiments/results/ablation_raw_<timestamp>.csv`.

**Three ablated variants:**
- `xtal-no-safety` — removes the SLA-pressure safety override (agent may pick conservative actions under pressure)
- `xtal-single-layer` — tunes only Flink knobs; Kafka linger always set to 0

---

## Week 6 — Paper

### Compile the paper

```bash
make paper
# Output: paper/main.pdf
```

Requires `pdflatex` and `bibtex`. On macOS: `brew install --cask mactex`

### Paper structure

| Section | File | Content |
|---------|------|---------|
| §1 Introduction | `sections/01_intro.tex` | Problem, gap, contributions |
| §2 Background | `sections/02_background.tex` | Flink, Kafka, XAI, contextual bandits |
| §3 Motivation | `sections/03_motivation.tex` | Baseline characterization: where spikes come from |
| §4 Design | `sections/04_design.tex` | State/action/reward, XGBoost bandit, SHAP, safety |
| §5 Implementation | `sections/05_implementation.tex` | Control loop overhead, API details |
| §6 Evaluation | `sections/06_evaluation.tex` | Results tables + figure refs |
| §7 Related Work | `sections/07_related.tex` | Prior art: autoscalers, RL, XAI, tracing |
| §8 Conclusion | `sections/08_conclusion.tex` | Summary + future work |

**To fill in the paper:**
1. Run `make eval` and `make ablation` to get real numbers
2. Run `make figures` to generate PDF figures
3. Replace placeholder numbers in `sections/06_evaluation.tex` with actual results
4. Add your Grafana screenshot exports as `paper/figures/` and update figure paths in the `.tex` files

---

## Docker Services Reference

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `xtal-kafka` | bitnami/kafka:3.6 | 9092 | Message ingest (KRaft, no ZooKeeper) |
| `xtal-kafka-exporter` | danielqsj/kafka-exporter | 9308 | Kafka metrics → Prometheus |
| `xtal-jobmanager` | apache/flink:1.18 | 8081, 9249 | Flink coordinator + REST API |
| `xtal-taskmanager-1` | apache/flink:1.18 | 9250 | Flink worker #1 |
| `xtal-taskmanager-2` | apache/flink:1.18 | 9251 | Flink worker #2 |
| `xtal-taskmanager-3` | apache/flink:1.18 | 9252 | Flink worker #3 |
| `xtal-prometheus` | prom/prometheus | 9090 | Metrics storage (7-day retention) |
| `xtal-grafana` | grafana/grafana | 3000 | Dashboard (admin / xtal2024) |

```bash
docker compose up -d           # start stack
docker compose down            # stop stack
docker compose down -v         # stop + wipe all volumes (fresh start)
docker compose logs -f kafka   # tail logs for a service
docker compose ps              # check health of all containers
```

---

## 6-Week Project Plan

| Week | Focus | Status | Key outputs |
|------|-------|--------|-------------|
| 1 | Infrastructure | ✅ Done | Docker stack, telemetry, actuator stubs, Grafana dashboard |
| 2 | Baseline Characterization | ✅ Done | `baseline_runner.py`, `failure_injector.py`, 4 failure modes |
| 3 | Control Plane Core | ✅ Done | `agent.py` (XGBoost bandit), `reward.py`, `main.py` (closed loop) |
| 4 | XAI + Integration | ✅ Done | 4 test files (unit + integration), `Makefile`, safety validation |
| 5 | Experiments | ✅ Done | `evaluate.py`, `ablation.py`, `plot_results.py` (6 paper figures) |
| 6 | Paper Writing | ✅ Done | Full LaTeX paper in `paper/` — 8 sections, bibliography, all figures |

---

## Paper Outline

1. **Introduction** — tail latency problem, gap in existing systems (1 page)
2. **Background** — Flink, Kafka, relevant prior work from the 19-paper survey (1.5 pages)
3. **Motivation** — Week 2 baseline data: where do tail spikes originate, how long is recovery (1.5 pages)
4. **XTAL Design** — control loop, state/action/reward, XGBoost bandit, SHAP layer, safety constraints (2.5 pages)
5. **Implementation** — system integration, control loop overhead measurement (1 page)
6. **Evaluation** — 3 baselines × 3 patterns × 3 runs, ablation, failure recovery (3 pages)
7. **Related Work** (0.5 pages)
8. **Conclusion** (0.5 pages)

**Target venues:** Middleware 2026, SoCC 2026

---

## Division of Work

| Component | Owner |
|-----------|-------|
| Infrastructure, Kafka/Flink instrumentation, actuator APIs, failure injection | Hrithik |
| Telemetry pipeline, XGBoost agent, SHAP integration, experiment automation, paper figures | Akanksha |
