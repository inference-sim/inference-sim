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

`metrics.Compare` returns the difference between two figures only when their definitions
are identical, field for field; otherwise it returns an error naming the first field that
differs. Identifiers decide nothing: the same definition under two names compares, and two
definitions under one name do not.

The simulator reads the registry directly. The dashboard and config search read it through
`blis metrics`.

## Definitions

--8<-- "docs/contracts/generated/metrics.md"

Source: [`metrics`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/metrics).
