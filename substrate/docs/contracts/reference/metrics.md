# Metric registry

<p class="lede">One definition, with an identifier, for every metric BLIS reports, compares,
ranks or uses in a service-level check (design §8.7, DEV-8). A Result names each figure by
its identifier; the definition lives here, once.</p>

A definition states the metric's formula (the per-sample quantity), its statistic, its
population (what the statistic is taken over), its window, its unit, and how it treats
requests unfinished at the horizon. Time to first token, time per output token and
inter-token latency follow the benchmark clients BLIS is compared with:

- `tpot_*` is per request: each completed request with at least two output tokens gives one
  sample, (E2E - TTFT) / (output tokens - 1), as `vllm bench serve` computes it.
- `itl_*` is pooled over tokens: every gap between consecutive output tokens of those same
  requests is one sample (inference-sim #1894).
- `step_interval_*` is the engine's step interval. Under speculative decoding one step
  emits several tokens, so it is not a per-token figure, and it is never compared with
  `itl_*` or `tpot_*`.

## Scope and aggregate

A figure is a metric at a scope (design §8.7): the whole cluster, one pool, or one instance.
The scope selects which requests, tokens or steps form the population; the formula, statistic,
window and unit do not change. A definition states the scopes it is defined at (`Scopes`), in
no particular order, and the aggregates its figures may take (`Aggregates`). Every metric is
defined at the cluster, a pool and an instance except the router's queue delay, which exists
only at the cluster and so has no aggregate: an aggregate ranges from a narrower scope to a
wider one, and a metric defined at one scope has nothing to range over.

A figure that picks one of the figures of a narrower scope names the scope it ranges over and
the aggregate: the largest p99 time to first token over instances is a figure at the cluster
scope, aggregate `max`, over `instance`. It is one instance's figure, not the cluster's own
p99, and the two are never compared.

The aggregates are `max` and `min`. Design §8.7 gives one example, the largest p99 time to
first token over instances, and the plan speaks of ranking on "the worst" instance. The worst
instance depends on which way the metric improves: the largest figure for a latency, the
smallest for a throughput, a hit rate or an attainment. Each definition therefore states its
direction (`Better`: `lower_is_better` or `higher_is_better`), and `Definition.Worst` returns
the aggregate that picks the worst instance for it. The design names no mean or sum of
figures, and an unweighted mean of per-instance percentiles or fractions is neither the
cluster's figure nor a weighted one, so neither is admitted. Adding one would be a design
change.

`Definition.Validate` refuses a definition with no scope, a scope or aggregate outside the
lists, a repeated one, no direction, and aggregates with fewer than two scopes to range
between. `Figure.Validate` first validates the figure's definition, then refuses a figure with
no scope, at a scope its metric is not defined at, with an aggregate its metric does not
admit, or ranging over a scope that is not narrower than its own.

## Comparing figures

`metrics.Compare` returns the difference between two figures only when `ComparableFigures`
accepts them: their definitions are identical, field for field, and their scope, member, aggregate
and the scope they range over are the same. Otherwise it returns an error naming the first field
that differs (`scope`, `member`, `aggregate` or `over` for the figure; the definition's field for the
definition). A figure at the pool or instance scope names its member, the pool or instance it
is of by the name the deployment gives it, and a figure at the cluster scope has none; design
§8.7 makes the scope "the whole cluster, one pool, or one instance", so instance 3's figure and
instance 7's are not comparable, and the same instance's in two runs is. For an aggregate, the
member is that of the figure's own scope: the largest p99 over the instances of the pool
`decode` is at the pool scope, member `decode`. Identifiers decide nothing: the same definition
under two names compares, and two definitions under one name do not. A figure with no scope is
refused.

The simulator reads the registry directly. The dashboard and config search read it through
`inference-sim metrics`.

## Definitions

--8<-- "docs/contracts/generated/metrics.md"

Source: [`metrics`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/metrics).
