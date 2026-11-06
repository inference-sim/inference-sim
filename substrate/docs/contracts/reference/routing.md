# Routing profiles

Expert-parallel routing imbalance: how unevenly a mixture-of-experts router loads the
ranks of an expert-parallel group. It depends on the model's router and on the traffic,
so, like [acceptance](acceptance.md), it is a profile keyed by model and workload
category, hosted under `routing/` in the registry, which a workload, a scenario or a
command line may override.

The quantity is vLLM's own. In v0.31.0 a step's load on rank r in layer l is the routed
(token, expert) assignments landing on r's experts
(`vllm/distributed/eplb/eplb_state.py:601-608`), and vLLM logs balancedness as
sum_l mean_r / sum_l max_r (`:66-71`). The token imbalance is its reciprocal, at least 1.
Each layer's combine waits for its slowest rank, so the layer pays the busiest rank's
routed compute. The expert imbalance is the busiest rank's distinct experts touched over
the per-rank count under uniform routing, which scales a weight-bound decode read and may
be below 1. Curves describe vLLM's default linear placement with EPLB off.

<!-- fields: routing.ProfileSet -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `RoutingProfileSet`. |
| `name` | string | yes | The set's identity, by convention its file name. |
| `role` | string | yes | `default`, consulted for every run its keys match, or `sensitivity`, used only when a scenario names it. |
| `caveats` | list of strings | no | How far the values travel. |
| `profiles` | list | yes | At least one profile, below. Two with one key are an error. |

<!-- fields: routing.Profile -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `model` | string | yes | The catalog model name. |
| `category` | string | yes | The workload category, spelled as for acceptance. |
| `phase` | string | yes | `decode` (tokens mixed across requests) or `prefill` (a chunk of one request). |
| `placement` | string | yes | `linear`, the only placement described. |
| `expert_parallel` | integer | yes | The width the curve is for. At least 2. |
| `points` | list | yes | The curve, by strictly increasing `batch_tokens`. |
| `method` | string | yes | One of the [methods](vocabularies.md#methods) except `not_charged` and `calibrated`. |
| `provenance` | mapping | unless `method` is `assumed` | Where the routes came from, below. |
| `rationale` | string | unless `method` is `measured` | The reasoning behind a value that was not measured. |

<!-- fields: routing.Point -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `batch_tokens` | integer | yes | Tokens in the step. A lookup interpolates in log(batch) and holds the end values outside the curve. |
| `token_imbalance` | number | yes | Busiest-rank routed tokens over the mean. At least 1. |
| `expert_imbalance` | number | yes | Busiest-rank distinct experts over the uniform-routing count. Positive. |
| `token_p10` | number | no | The step-to-step 10th percentile of the token imbalance. |
| `token_p90` | number | no | The 90th percentile. Not below `token_p10`. |

<!-- fields: routing.Provenance -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `source` | string | yes | The dataset the routes were read from, with its revision. |
| `derivation` | string | no | The script and output digest. |
| `hardware` | string | no | The chip the routes were captured on. A resolution on other hardware records it. |
| `engine` | string | no | The engine build. |
| `requests` | integer | no | How many requests the routes cover. |
| `tier` | integer | no | The source's reputation tier, 1 or 2. |

## Inline values

A workload's `routing`, a scenario's `routing.value` and a command line's value. Flat in
the step size.

<!-- fields: routing.Value -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `token_imbalance` | number | yes | At least 1. 1 is evenly loaded. |
| `expert_imbalance` | number | no | Positive. Omitted, 1. |
| `source` | string | no | Where the value came from, recorded with the resolution. |

## Scenario pin

<!-- fields: routing.Pin -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `profiles` | string | one of two | A routing-profile set under `routing/` in the registry, default or sensitivity. A set without the run's key is refused. |
| `value` | mapping | one of two | An inline value, above. |
| `fallback_category` | string | no | With `profiles`, the category whose curves to use when the workload's has none. The substitution is recorded. Not with `value`. |

## Resolution

`routing.Resolve` picks the highest layer that states an imbalance: the command line, the
scenario's pin, the workload's value, then the registry's `default`-role sets for (model,
category, phase, linear, width). With none, or without expert parallelism, or with EPLB
on and no pin, the run is priced as evenly loaded and the resolution records why. A
registry miss is not an error; a scenario's named set that misses is.
