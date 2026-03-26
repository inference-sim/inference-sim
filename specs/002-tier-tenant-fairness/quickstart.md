# Quickstart: Phase 1B — Service Tiers & Tenant Fairness

## Tier-Ordered Load Shedding

Enable tier-ordered shedding by setting the admission policy to `tier-shed`:

```bash
# Via CLI flag
./blis run --model qwen/qwen3-14b --admission-policy tier-shed

# With overload threshold (shed Sheddable when any instance queue > 5)
./blis run --model qwen/qwen3-14b \
  --admission-policy tier-shed \
  --tier-shed-threshold 5
```

Or in a policy bundle YAML:

```yaml
admission:
  policy: tier-shed
  tier_shed_threshold: 5     # shed when max effective load > 5 (default: 0)
  tier_shed_min_priority: 3  # shed tiers below Standard (default: 3 = Standard and above pass)
```

**What happens:**
- `critical` and `standard` requests: always admitted (unless system is at absolute capacity)
- `sheddable` requests: shed when max instance effective load > `tier_shed_threshold`
- `batch` and `background` requests: deferred to idle-capacity queue (see below)

## Deferred Queue (Batch/Background)

No configuration needed — automatic when `batch` or `background` SLO tier is present in the workload.

```yaml
# workload spec v2 — mix of real-time and batch traffic
clients:
  - id: chat-users
    slo_class: standard
    rate: 10
  - id: eval-jobs
    slo_class: batch      # these go to the deferred queue under load
    rate: 5
```

**What happens:**
- Batch requests wait in a deferred queue while the cluster is serving real-time traffic
- They are released and processed as soon as all instance wait queues become empty
- At simulation end, any unprocessed deferred requests appear as `deferred_horizon_interrupted` in the anomaly counters (satisfying INV-1 request conservation)

## Per-Tenant Fair-Share

Configure tenant budgets in the deployment config:

```yaml
# deployment.yaml
num_instances: 4
tenant_budgets:
  "tenant-a": 0.40   # tenant-a gets up to 40% of cluster capacity
  "tenant-b": 0.60   # tenant-b gets up to 60%
```

```bash
./blis run --model qwen/qwen3-14b --deployment deployment.yaml
```

**What happens:**
- When `tenant-a` holds > 40% of in-flight slots, their `sheddable`, `batch`, and `background` requests are rejected preferentially
- `critical` and `standard` requests from over-budget tenants are still admitted
- Tenants with no budget configured are unlimited

## Reading the Output

After a simulation with mixed tiers and multiple tenants:

```
=== Per-Tenant Metrics ===
tenant-a: completed=420 tokens=84000
tenant-b: completed=580 tokens=116000
Jain Fairness Index: 0.992

=== Anomaly Counters ===
Rejected Requests (Admission): 150   ← includes tier-shed + tenant-budget rejections
Deferred Horizon-Interrupted: 12     ← batch/background not processed before horizon
```

## Checking Invariants

```bash
# Verify monotonic shedding order (SC-001)
./blis run --model qwen/qwen3-14b --admission-policy tier-shed \
  --rate 200 --num-requests 1000 | grep "shed"

# Verify INV-1 conservation with deferred queue
# injected == completed + running + queued + shed + dropped + deferred_horizon_interrupted
```
