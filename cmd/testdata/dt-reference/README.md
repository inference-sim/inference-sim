# DT adapter-fidelity reference fixtures (PR7 / #1470)

Committed fixtures for the BLIS-vs-Digital-Twin adapter-cost fidelity comparison
(`blis calibrate --adapter-reference`) and the SC-007 test
(`TestSC007_AdapterFidelity_Fixtures`).

Per config (`qwen-2.5-7b-instruct`, `llama-3.1-8b-instruct`):

| File | Source |
|---|---|
| `<config>.dt.json` | Agullo Digital Twin `simulate()` aggregates (adapter-aware + adapter-blind), from the `GPULLMAdapterOptimization` fork (arXiv:2508.08343) via `lora-control/experiments/export_dt_reference.py`. |
| `<config>.blis-aware.json` | BLIS aggregate MetricsOutput, adapter physics ON (`--lora-config`). |
| `<config>.blis-blind.json` | BLIS aggregate MetricsOutput, adapters inert (no `--lora-config`), same arrivals. |

## Workload

Both sides run the **identical** seeded arrival stream (the DT's canonical
`dt_driver.py` workload: 16 adapters, ranks {8,16,32}, 8 slots, 0.5 req/s each,
60 s, `np.random.default_rng(0)`). BLIS replays the DT's exact arrivals via a
TraceV2 built from the exported stream, so any error is attributable to model /
scheduler / adapter-physics differences, not to re-drawn arrivals. The DT
loading + overhead coefficients are mapped to `LoRAConfig` (`k6/k7 =
slope/intercept`, rank-blind, matching the DT `_small` overhead variant).

## Regenerate

```bash
# 1. DT references (needs the fork; GPULLM_REPO points at it)
GPULLM_REPO=/path/to/GPULLMAdapterOptimization \
  python3 lora-control/experiments/export_dt_reference.py /tmp/dt-ref-out
# 2. Build a BLIS TraceV2 + LoRAConfig from the exported arrival stream, then
#    replay it twice (aware = with --lora-config, blind = without) and capture the
#    aggregate MetricsOutput. `blis replay` intentionally exposes only
#    --results-path (BC-2), so read the "=== Simulation Metrics ===" JSON block
#    from replay's stdout and trim to the aggregate fields above.
```

## Producing `--sim-metrics` yourself

The general user flow for the comparison uses `blis run`, which writes a clean
aggregate MetricsOutput via `--metrics-path`:

```bash
blis run --model qwen2.5-7b-instruct --hardware H100 --tp 1 \
  --workload-spec <your-adapter-workload>.yaml --lora-config <dt-coeffs>.yaml \
  --metrics-path aware.json
blis calibrate --adapter-reference qwen-2.5-7b-instruct.dt.json \
  --sim-metrics aware.json --report report.json
```

## Finding (SC-007, design §15 falsification path)

| Metric | qwen absMAPE | llama absMAPE | ≤ 20%? |
|---|--:|--:|:--:|
| **throughput** (compute-overhead term, D4) | 14.2% | 14.4% | ✅ validated |
| **TTFT** | 98.5% | 97.2% | ❌ **unsupported** |

TTFT diverges because BLIS ports adapter *deltas* onto its own separately-calibrated
base (design §6), which is ~100× faster than the DT's H100 base fit (BLIS blind
TTFT ≈26 ms vs DT ≈3250 ms); the DT's large absolute TTFT is queueing amplification
BLIS's faster base never enters. TTFT is also the DT's own weakest axis (~17–21%
SMAPE vs real; throughput ~5%, §15). The comparison mechanism ships regardless; the
TTFT fidelity claim is withheld, not silently passed.

**Caveat (workload):** the canonical DT workload does not saturate BLIS, so the
throughput agreement partly reflects both systems tracking offered load (BLIS
aware/blind throughput ratio ≈0.999 vs DT ≈0.90). A saturating-workload validation
of the D4 compute-overhead term is future work.

**Caveat (rank-blind coefficients):** the DT overhead fit used here
(`predictor_latency_overhead_small`) is linear in the unique-adapter count and
**ignores rank**, so the LoRAConfig coefficients are mapped rank-blind
(`k6/k7 = slope/intercept`, identical across the {8,16,32} rank tiers). This
comparison therefore does **not** exercise BLIS's rank-tiered overhead
(FR-009 / D4's max-rank-in-batch term) — with a single rank per adapter the tier
key is fixed. Validating the rank dependence would require a rank-dependent DT fit
(the `_small` variant's commented-out sigmoid, or a re-profile); BLIS's rank
tiering is instead covered by the PR4 rank/uniqueness-sensitivity unit test.
