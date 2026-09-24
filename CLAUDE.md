# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Charter — read before adding anything here.** CLAUDE.md is agent *operating context*, not a
> changelog. It is loaded into every session, so it stays small. Per-PR rationale, "behavior
> change" notes, migration tables, and byte-identity evidence go in the **commit body** and the
> relevant **`docs/` guide** — never here. Add to CLAUDE.md only a durable fact an agent needs
> *every* session that has no more-authoritative home; when a fact has a canonical doc, CLAUDE.md
> carries a **pointer, not a copy**. A CI job enforces a size ceiling on this file.

## Project Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — driven by trained performance coefficients (alpha/beta), analytical roofline estimates, or physics-informed cross-model prediction.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Catalog and deployment rules (mandatory)

Every `blis run` / `blis replay` obeys these; see `docs/getting-started/` and `docs/reference/configuration.md` for detail.

1. **Locate the catalog** with `--catalog <path>` or the `BLIS_CATALOG` env var — no default, no search path, no remote fetch (`--catalog` wins, announced on stderr; neither is refused). Both name the catalog **clone root**. Clone `blis-catalog` at its pinned release tag and export it once (`docs/getting-started/installation.md#catalog-compatibility` is canonical):
   ```bash
   git clone --branch 0.1.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
   export BLIS_CATALOG=$PWD/blis-catalog
   ```
2. **A model runs iff it is in the catalog.** Its config is read from `<catalog>/models/<short-name>/config.json`; an absent entry is refused naming that path. No run-time HuggingFace fetch; no command creates or modifies a catalog file. Add a model by committing its config to `blis-catalog`.
3. **`--hardware` and `--tp` are required** on both `run` and `replay` (refused naming the missing flag). `blis observe` takes neither (it resolves no model config).
4. **Named workload presets** come from `<catalog>/workloads/<name>.yaml` — used by `run --workload`, `observe --workload`, and `convert preset --name` (all take `--catalog`/`BLIS_CATALOG`).

## Build and Run Commands

One canonical form per command. `--help` and the guides carry the full flag surface — by home:

- SLO goodput, flow-control, dispatch ordering → `docs/guide/admission.md`
- speculative decoding/MTP, MoE `--dp`/`--enable-expert-parallel`, `--kv-cache-dtype`, inter-node cost → `docs/guide/latency-models.md` (+ `docs/reference/models.md` for MLA/hybrid/quant)
- KV-offload (`--kv-offload-config`, `--scrape-kv-metrics`) → `docs/guide/kv-offload-calibration.md`
- closed-loop/corpus sessions, `observe`, `calibrate` → `docs/guide/observe-replay-calibrate.md`
- workloads & `convert` → `docs/guide/workloads.md`
- saturation detectors → `docs/contributing/saturation-analyzer-extension.md`

```bash
# Build
go build -o blis main.go

# Run (single deployment). --hardware and --tp required; catalog located via BLIS_CATALOG.
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1

# Run and export the workload as a TraceV2 (prefix auto-appends .yaml/.csv)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --trace-output traces/run1

# Replay a captured TraceV2 through the DES (identical flags reproduce run metrics — INV-13)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b --hardware H100 --tp 1

# Observe a real server's latency into a TraceV2 (black-box dispatcher; no --hardware/--tp)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --trace-header trace.yaml --trace-data trace.csv

# Compare real observed latencies against simulator predictions
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json --report calibration.json

# Convert a workload/agentic trace to a TraceV2 (preset | servegen | inference-perf | otel | weka)
./blis convert preset --name chatbot --rate 10 --num-requests 100
```

## Testing

```bash
go test ./...                      # all tests
go test ./sim/...                  # a package tree
go test ./sim/... -run TestKVCache # a single test
go test -cover ./...               # with coverage
golangci-lint run ./...            # lint (v2.9.0); run before pushing
```

## Development Guidelines

### Design Principles

BLIS follows a layered design-document hierarchy (read the design guidelines first when designing anything new):

- **Design guidelines** (`docs/contributing/templates/design-guidelines.md`): target architecture, DES foundations, module contracts, extension framework.
- **Design docs** (per-feature): behavioral specs — what modules do and why, never how.
- **RFC + `.archon` plan** (multi-PR features): tracking issue with holes/surfaces/contracts (`docs/contributing/rfc.md`), encoded into a machine-checkable plan.
- **Implementation plans** (single PR): behavioral contracts + TDD tasks — follow `docs/contributing/pr-workflow.md`.

**Module architecture:** a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) plus domain-specific modules (router, scheduler, KV cache manager, latency model, autoscaler, batch formation). Each module has a six-aspect behavioral contract; see design guidelines §4–§5 for the module map and the four extension recipes.

### BDD/TDD, PR Workflow, Standards — pointers (canonical sources)

- **BDD/TDD:** write behavioral contracts first, tests before code, table-driven tests, test laws not just values, and prefer tests that survive a rewrite. Canonical: `docs/contributing/standards/principles.md` (BDD/TDD section).
- **PR workflow:** follow `docs/contributing/pr-workflow.md` (worktree → source audit → plan → review → implement → self-audit → commit). Before approving a plan, validate it: task dependencies are startable, all template sections present, the executive summary is clear to a newcomer, and no task is under-specified. An RFC (`docs/contributing/rfc.md`) precedes implementation for changes that introduce/modify module boundaries; smaller changes go straight to the PR workflow.
- **Code review:** check antipattern rules R1–R23 — `docs/contributing/standards/rules.md`. Watch R8–R10 (exported mutable maps, YAML pointer types, strict config parsing). Always run `go test ./...` and lint after fixes.
- **Engineering principles** (`docs/contributing/standards/principles.md`): `sim/` is a library (never terminates); dependency direction `cmd/ → sim/cluster/ → sim/`; single-method interfaces with factory validation; config grouped by module (R16); canonical constructors in one place (R4); **stdout = deterministic results, stderr = diagnostics**; CLI errors → `logrus.Fatalf`, library → `error`/`panic`, never a silent `continue` (R1).

### Key Invariants

Full registry (19 core invariants INV-1…INV-19 plus INV-A, INV-BC-DP1, the LoRA/PD/pool families, and NS-6), with verification strategies: **`docs/contributing/standards/invariants.md`** — the canonical source; every `INV-*` cited in `sim/`/`cmd/` resolves there. The ones you touch most:

- **INV-1 Request conservation** — injected == completed + all still-in-flight/dropped/rejected terms at end (twelve-term cluster form; five-term single-instance form).
- **INV-6 Determinism** — same seed ⇒ byte-identical **stdout**; wall-clock timing goes to stderr.
- **INV-13 Run/Replay parity** — a trace exported by `run` and replayed with identical flags (including `--horizon`) yields byte-identical stdout; unsupported replay features `logrus.Fatalf` at startup.
- **NS-6 Catalog is authoritative and read-only** — see *Catalog and deployment rules* above; binds the CLI only.

### Notable subsystems (pointers)

Routing scorers + the default profile (`precise-prefix-cache:2,queue-depth:1,kv-utilization:1`): `docs/guide/routing.md`. LoRA control plane, MoE `--dp` placement + expert parallelism, speculative decoding/MTP, tiered KV-offload, and per-detector saturation live in the guides mapped above and `docs/reference/*` (and `git log` for history). Extension recipes for policies, scorers, latency backends, KV tiers, trace records, and metrics: `docs/contributing/extension-recipes.md`.

### Code Style

- Composition over inheritance (e.g. `InstanceSimulator` wraps existing `sim` components).
- Timestamp-based event ordering via min-heap; cluster and per-instance queues use `(timestamp, priority, seqID)`; cluster ties broken by lowest instance index.
- Partitioned RNG per subsystem to isolate randomness.

### CI/CD

GitHub Actions run on all PRs to main:

- `.github/workflows/ci.yml` — build (`go build ./...`), lint (`golangci-lint`, v2.9.0), tests, and a CLAUDE.md size-ceiling check.
- `.github/workflows/docs.yml` — MkDocs site (PR build-only; deploy on push to main; versioned on tag).

## Agent Behavioral Instructions

The following are for Claude Code and other AI assistants. Human contributors can skip this section.

### GitHub Action: PR Reviews

When triggered via `@claude /blis-pr-review` on a PR, follow the blis-pr-review skill exactly. For all other triggers (questions, debugging, etc.), respond normally without creating a PR unless explicitly asked.

The `/blis-pr-review` path runs with a **read-only** token (`contents: read`, #1697): it can post the review comment but cannot push. Report findings — do not fix them, and do not attempt a commit or push (it fails with 403). Every other `@claude` trigger keeps `contents: write`. See `docs/contributing/standards/agent-trust.md`.

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize concisely. Never synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental per-agent summaries rather than a consolidated report.

### Task Agent Guidelines

1) Do NOT poll repeatedly — check at reasonable intervals (every 30–60s, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.

### Issue Filing

<!-- Keep in sync with .github/ISSUE_TEMPLATE/ — update when templates change -->

Pick the template that matches your situation, reproduce its structure, and apply its front-matter labels (every issue needs at least one label):

1. Bug or wrong simulation result → `Bug report` (`.github/ISSUE_TEMPLATE/bug_report.md`)
2. Porting a feature from an external repo (llmd, gaie, vllm, sglang) → `Cross-repo feature` (`cross_repo_feature.md`) — requires source permalinks
3. New BLIS-native capability → `Feature request` (`feature_request.md`)
4. Hypothesis or experiment → `Hypothesis Proposal` (`hypothesis.md`)
5. Antipattern / hardening / refactoring → `Hardening / refactoring` (`custom.md`)

## Latency Estimation

Two modes, selected via `--latency-model`. **Trained-physics is the default** (roofline basis functions plus learned correction coefficients; generalizes across architectures, workloads, and TP configs — no per-model calibration). **Roofline** (`--latency-model roofline`) is a pure analytical FLOPs/bandwidth model. The deprecated `blackbox`/`crossmodel`/`trained-roofline` backends were removed. Detail (MoE/EP, MLA, hybrid attention, spec-decode, quantization, KV dtype, inter-node network cost): `docs/guide/latency-models.md` and `docs/reference/models.md`.

## Post-Hoc Saturation Detection

Analyzes completed runs (distinct from the real-time flow-control detector). Package `sim/saturation/`; streaming detectors `composite`, `threshold`, `backlog-drift`, `peak-rate`, each with a false-alarm calibration knob. CLI (on `run`, `replay`, `observe`): `--detectors <name|comma-list|all>`, `--saturation-config <yaml>`, `--saturation-report <path>` (writes `{"final":{...},"trace":[...]}`), `--saturation-final-window <duration>`. run→replay is byte-identical (INV-13). Full mechanism, tuning knobs, and migration from the pre-#1516 flags: `docs/contributing/saturation-analyzer-extension.md`.

## File Organization

For the full annotated file tree, see `docs/reference/project-structure.md`.

## Key Data Flow

Request pipeline: Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion. Admission and Routing apply in cluster mode only; single-instance skips to WaitQueue. Full diagram: `docs/concepts/architecture.md`.

## Project Governance Documents

### Standards (what rules apply)

- `docs/contributing/standards/rules.md` — 23 antipattern rules (R1–R23), each with evidence and enforcement.
- `docs/contributing/standards/invariants.md` — the invariant registry (see *Key Invariants* above).
- `docs/contributing/standards/principles.md` — engineering principles; also the source-of-truth map for where each fact is canonical.
- `docs/contributing/standards/agent-trust.md` — agent trust boundaries (Trusted / Verify-after / Never-trust).

### Process (how to do each activity)

- `docs/contributing/pr-workflow.md` — end-to-end PR workflow. **Step 1.5:** the source document is the issue body **plus** the design refinements in its comment thread; `scripts/deliver-issue-refinements.sh <issue>` prints the comments that carry authority (author holds `admin`/`write`/`maintain`; a refinement overrides the body; later refinements win; target branch / `archon-plan:` / `Depends on:` stay body-only).
- `docs/contributing/issue-comment-authority.md` — decision record for that rule.
- `docs/contributing/automated-delivery.md` — the L1 `/approve-issue-for-pr-delivery` loop.
- `docs/contributing/rfc.md`, `docs/contributing/templates/rfc-to-plan.md` — RFC template and `.archon` encoding.

### Templates & Plans

- `docs/contributing/templates/design-guidelines.md` — start here when designing anything new.
- Active plans: `docs/plans/` (gitignored). Archived design docs: `docs/plans/archive/`.

## Active Technologies

- Go 1.22+ with `gopkg.in/yaml.v3` (strict parsing), `gonum` (stats), `cobra`, `logrus`.
- In-memory node/GPU inventory maps; no external storage.

## Change History

See `git log --oneline main` for the definitive commit history. Durable cross-cutting facts live in the standards docs and topic guides (admission, routing, scheduling, observe-replay-calibrate, workloads, latency models, KV offload, saturation, configuration reference).
