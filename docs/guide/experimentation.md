# Hypothesis-Driven Experimentation

This guide covers using BLIS as a platform for rigorous, reproducible experiments. Because BLIS is a deterministic discrete-event simulator (same seed → byte-identical output), you can run controlled experiments that are impossible with real hardware.

```bash
# Quick example: compare chunked prefill vs no chunked prefill
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --rate 100 --num-requests 500 --long-prefill-token-threshold 0

./blis run --model meta-llama/llama-3.1-8b-instruct \
  --rate 100 --num-requests 500 --long-prefill-token-threshold 256
```

## Why Experiment with a Simulator?

Real GPU benchmarks suffer from noise: wall-clock jitter, OS scheduling, GPU thermal throttling, network variability. With BLIS:

- **Deterministic replay** — change exactly one variable, attribute all output differences to that change
- **No hardware cost** — run thousands of configurations on a laptop
- **Controlled conditions** — isolate the effect of a single parameter while holding everything else constant
- **Reproducible** — share your seed, workload spec, and CLI flags; anyone can reproduce your results

## Capacity Planning Validation

The most common experiment workflow for platform engineers:

1. **Define your deployment:** model, GPU, TP, instance count
2. **Define your workload:** arrival rate, token distributions (from production logs if available)
3. **Define your SLO:** TTFT p99 < 200ms, E2E p99 < 5s, etc.
4. **Run the simulation** with these parameters
5. **Interpret:** Does the simulated TTFT p99 meet your SLO? If not, add instances or tune routing.

```bash
# Example: Will 8 instances of LLaMA 3.1 8B handle 400 req/s at TTFT p99 < 500ms?
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 8 --rate 400 --num-requests 2000 \
  --routing-policy weighted --seed 42
```

## The `/hypothesis-experiment` Skill

For structured hypothesis-driven research, BLIS includes a guided experimentation workflow via the `/hypothesis-experiment` Claude Code skill:

```
/hypothesis-experiment
```

This skill guides you through:

1. **Formulate** — state a testable prediction (e.g., "chunked prefill reduces short-request TTFT p99 by > 30%")
2. **Classify** — identify the hypothesis family (scheduler invariants, performance-regime, etc.)
3. **Design** — specify parameters, controls, success criteria
4. **Implement** — create `run.sh` (experiment script) and `analyze.py` (analysis script)
5. **Run** — execute the experiment
6. **Analyze** — parse results, compute statistics
7. **Document** — write FINDINGS.md with conclusions and evidence

## The Experiment Harness

All experiments use a shared harness (`hypotheses/lib/`) for consistency:

```bash
source hypotheses/lib/harness.sh

# Run a simulation with standard setup
blis_run 60 results/baseline.json \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500
```

The harness provides:
- `blis_run()` — wrapper around the simulation binary
- `setup_experiment()` — create output directories
- `preflight_kv_check()` — verify KV configuration
- `hypotheses/lib/analyze_helpers.py` — common analysis functions (`parse_blis_output()`, etc.)

## Case Studies

Completed experiments demonstrate the power of hypothesis-driven analysis:

| Experiment | Finding | Impact |
|-----------|---------|--------|
| **H7 (Horizontal Scaling)** | TTFT p99 scales 7.4x (not 2x) when doubling instances near saturation | Super-linear benefit from queue growth rate reduction |
| **H27 (Chunked Prefill)** | `--long-prefill-token-threshold=256` reduces short-request TTFT p99 by 52% | But ITL is unaffected — chunked prefill benefits scheduling, not decode |
| **H29 (Snapshot Staleness)** | `--snapshot-refresh-interval` 100ms degrades TTFT p99 by +354% for kv-utilization scorer | Safe zone < 5ms; composite scorer mitigates ~99% |
| **H20 (Heavy-Tailed)** | ParetoLogNormal produces fewer preemptions than Gaussian despite similar means | Distribution median, not mean, drives KV pressure |

All findings are documented in `hypotheses/*/FINDINGS.md`.

## Convergence Review

Experiments go through a multi-perspective review process to ensure rigor:

1. **Design Review** (5 perspectives) — validates hypothesis quality and experiment design
2. **Code Review** (5 perspectives) — checks run.sh/analyze.py for correctness
3. **FINDINGS Review** (10 perspectives) — validates conclusions against evidence

The `/convergence-review` skill automates this process. Zero CRITICAL + zero IMPORTANT findings = converged.

## Further Reading

- [Hypothesis Process](../contributing/hypothesis.md) — full 10-step process for contributors
- [Experiment Standards](../contributing/standards/experiments.md) — rigor requirements (ED-1 through ED-6, RCV-1 through RCV-6)
- [Metrics & Results](results.md) — understanding the metrics your experiments produce
