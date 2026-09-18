# Quickstart: Tier-Ordered Admission Shedding

## Enable Tier-Ordered Shedding

Set the admission policy in your deployment config:

```yaml
# deployment.yaml
num_instances: 4
admission_policy: tier-shed
tier_shed_threshold: 5      # start shedding when any instance has effective load > 5
tier_shed_min_priority: 3   # shed tiers below Standard (default: 3)
```

```bash
./blis run --model qwen/qwen3-14b --deployment deployment.yaml
```

Or as a workload scenario in your YAML spec alongside existing config.

## Behavior at a Glance

With `tier-shed` admission active:

| SLO Tier | Under load ≤ threshold | Under load > threshold |
|----------|------------------------|------------------------|
| critical | admitted | admitted |
| standard | admitted | admitted (priority 3 ≥ min 3) |
| sheddable | admitted | **rejected** |
| batch | admitted | admitted (pass-through for deferred queue) |
| background | admitted | admitted (pass-through for deferred queue) |

`tier_shed_threshold: 0` (default) means shedding starts immediately when any instance has any effective load.

## Verify Monotonic Shedding

```bash
# Run a five-tier workload at 2x capacity
./blis run --model qwen/qwen3-14b --deployment deployment.yaml \
  --num-requests 1000 --rate 200

# Check output: shed(Sheddable) >= shed(Standard) >= shed(Critical)
# Per-tier rejection counts appear under "Rejected Requests" in anomaly counters
```

## Zero-Regression Check

Simulations not using `tier-shed` are unaffected:

```bash
# Run baseline (no tier-shed)
./blis run --model qwen/qwen3-14b --seed 42 > baseline.txt

# Run with any other admission policy (or default always-admit)
./blis run --model qwen/qwen3-14b --seed 42 > current.txt

diff baseline.txt current.txt  # must be empty
```
