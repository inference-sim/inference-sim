# Quickstart: Per-Tenant Jain Fairness Index

## What it does

After any simulation run or trace replay involving multi-tenant workloads, BLIS now prints a `=== Per-Tenant Metrics ===` section showing how equitably the cluster served each tenant. The Jain fairness index (0.0–1.0, where 1.0 = perfectly fair) gives a single-number equity signal.

## Triggering the output

The section appears automatically when any request in the workload has a non-empty `tenant_id`. No new flags or config required.

### Via workload spec

```yaml
clients:
  - id: alice-client
    tenant_id: alice
    slo_class: standard
    rate_fraction: 0.5
    prompt_tokens: 512
    output_tokens: 128

  - id: bob-client
    tenant_id: bob
    slo_class: standard
    rate_fraction: 0.5
    prompt_tokens: 512
    output_tokens: 128
```

```bash
./blis run --model qwen/qwen3-14b --workload-spec workload.yaml
```

### Via named preset (chatbot has two tenants by default)

```bash
./blis run --model qwen/qwen3-14b --workload chatbot --rate 10 --num-requests 100
```

### Via trace replay

```bash
./blis replay --trace-header trace.yaml --trace-data trace.csv --model qwen/qwen3-14b
```

## Reading the output

```
=== Per-Tenant Metrics ===
  alice: requests=50, tokens=12500
  bob:   requests=50, tokens=12480
  Jain Fairness Index: 0.9999
```

| Value | Meaning |
|-------|---------|
| `requests=50` | 50 completed requests attributed to this tenant |
| `tokens=12500` | Total output tokens served to this tenant |
| Jain ≈ 1.0 | Near-perfect fairness — both tenants received equal service |
| Jain < 0.70 | Severe imbalance — one tenant received significantly more service |

## When the section is absent

If all requests have empty `tenant_id` (e.g., legacy single-tenant workloads), the section does not appear. This is by design — no spurious empty sections for backward-compatible workloads.
