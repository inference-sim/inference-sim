# Quickstart — LoRA Placement-Policy Seams

Illustrative usage once B-1…B-7 land. Flag names are indicative (finalized per micro-plan).
Every command with no placement options is byte-identical to today (INV-6).

## 1. Reproduce a static placement (US1 — AQ1)

Pre-assign adapters to instances and route strictly to holders:

```bash
./blis run --model qwen/qwen3-14b \
  --lora-config lora.yaml \            # adapter registry + capacity + cost coeffs
  --creation-policy pre-placement \    # seed the declared adapter→instance assignment at t=0
  --routing-policy route-to-holder     # serve each adapter only from a holder
```

`lora.yaml` (deployment/cluster scope) declares the adapter→instance assignment. Expected:
pre-placed adapters resident at t=0 with **zero cold-loads**; 100% of their requests served by a
holder (SC-002). The effective triple is recorded in the run's `MetricsOutput`.

## 2. Eviction ablation under skew (US2 — AQ2)

Same workload, swap only the eviction policy, hold routing/creation fixed and seed fixed:

```bash
./blis run --model qwen/qwen3-14b --lora-config lora.yaml \
  --eviction-policy lru            # baseline
./blis run --model qwen/qwen3-14b --lora-config lora.yaml \
  --eviction-policy rank-aware     # cost-aware victim selection
```

Under skewed adapter popularity with differing ranks, the rank-aware run's victim tracks the
reload-cost criterion (not merely "differs from LRU"), changing per-adapter load/eviction counts
while routing/creation behavior is unchanged (SC-003, orthogonality).

## 3. Select a whole policy by bundle name (US3)

```bash
./blis run --model qwen/qwen3-14b --lora-config lora.yaml \
  --lora-bundle toppings                 # expands to a {routing, eviction, creation} triple
./blis run --model qwen/qwen3-14b --lora-config lora.yaml \
  --lora-bundle toppings --eviction-policy lru   # override just the eviction knob
```

The run output records the **effective** post-expansion triple (SC-006), so the result is
reproducible from the record alone.

## 4. Confirm the no-op default (US4 — the safety check)

```bash
./blis run --model qwen/qwen3-14b --seed 42        # no placement options
```

Byte-identical to the pre-feature golden for the same seed/config (SC-001). Setting every seam
explicitly to baseline (`--eviction-policy lru --creation-policy on-demand` + existing routing
profile) produces the same output.

## Notes
- **Paired comparisons (AQ1/AQ2)**: strict common-random-numbers validity needs RNG-free router
  tie-breaking, which no flag exposes yet (§14 follow-up) — for now, comparisons carry the documented
  CRN caveat.
- **Autoscaler**: pre-placement composes with `--model-autoscaler-interval-us`; a scaled-in instance
  starts empty and admits on-miss (cross-scale re-placement is out of scope).
- **Replay**: `blis replay` accepts the same policy selection; export→replay yields identical
  per-request metrics (INV-13).
