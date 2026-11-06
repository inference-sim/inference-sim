# Acceptance profiles

<p class="lede">How far speculative decoding advances a request per verification step. The
registry holds measured acceptance lengths with their provenance; a workload, a scenario
or the command line may state their own; and a run resolves exactly one of them and
records where it came from.</p>

## The quantity

The canonical value is the **mean acceptance length** (AL): the tokens a request gains per
verification step, counting the target's bonus token. For K speculative tokens it lies in
[1, K + 1]. It is what InferenceX's golden curves publish and what the engines take
(vLLM `synthetic_acceptance_length`, SGLang `SGLANG_SIMULATE_ACC_LEN`).

Two other quantities are derived from it, and each has one name here:

- The **acceptance rate** is the accepted fraction of drafts, (AL − 1) / K. It is vLLM's
  logged "Avg Draft acceptance rate" and inference-sim's `--speculative-acceptance-rate`.
  The two agree only when every step drafts all K: with a drafter that verifies fewer on
  some steps, such as DSpark under `enable_adaptive_verification`, vLLM's logged rate
  (accepted over drafted tokens, `vllm/v1/spec_decode/metrics.py:107-110`) divides by the
  drafts actually proposed and diverges from (AL − 1) / K.
  The per-token probability p of the speculative-decoding papers is a different quantity
  and is never called a rate in this schema.
- **Per-position rates** p<sub>1</sub>..p<sub>K</sub> follow vLLM's
  `synthetic_acceptance_rates`: p<sub>i</sub> is the unconditional probability that the
  first i drafts are all accepted. They are non-increasing, each lies in [0, 1], and they
  sum to AL − 1.

No source in scope publishes per-position rates, so they are optional everywhere. A
consumer that needs a distribution derives one with `MinimumVarianceRates`, which is
vLLM v0.31.0's `_acceptance_length_to_rates` (`vllm/config/speculative.py:557-564`):

```
rates = [1.0] * floor(AL - 1) + [frac(AL - 1)] + [0.0] * rest, truncated to K
```

This is how vLLM, and the InferenceX runs that force a published AL, realise one, so a
simulator using it reproduces those runs. Natural acceptance spreads more than this
schedule does, so per-request tails derived from it are optimistic.

Acceptance is not the kernel's input. K and the method stay in the deployment's
[`speculative`](deployment.md) block, because they change what a step costs; acceptance
changes only how far requests advance.

## Profile set

A registry file under `acceptance/`, named by its file. This one carries real values from
InferenceX's golden SPEED-Bench curves:

```yaml
--8<-- "docs/contracts/examples/acceptance-profile-set.yaml"
```

<!-- fields: acceptance.ProfileSet -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `AcceptanceProfileSet`. |
| `name` | string | yes | The set's identity. By convention it equals the file name, which is what a scenario's `acceptance.profiles` uses. |
| `caveats` | list of strings | no | What the set does not cover and how far its values travel. No entry may be empty. |
| `profiles` | list | yes | At least one profile, below. |

### Profiles

A profile is keyed by model, speculative method, K, thinking mode and workload category.
Two profiles with one key are an error.

<!-- fields: acceptance.Profile -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `model` | string | yes | The catalog model name, not the source's spelling of it. |
| `speculative_method` | string | yes | The engine's method name, as a deployment's `engine.speculative.method` states it. |
| `num_spec_tokens` | integer | yes | K. At least 1. |
| `thinking` | string | yes | `on` or `off`. One of the [thinking modes](vocabularies.md#thinking-modes). |
| `category` | string | yes | The workload category, such as `coding`. Lower-case letters, digits, `_` and `-`. |
| `acceptance_length` | number | yes | AL, in [1, K + 1]. |
| `per_position_rates` | list of numbers | no | K rates, each in [0, 1], non-increasing, summing to AL − 1 within 0.01. |
| `method` | string | yes | How the value was obtained. One of the [methods](vocabularies.md#methods) except `not_charged`. |
| `provenance` | mapping | unless `method` is `assumed` | Where it was measured, below. |
| `rationale` | string | unless `method` is `measured` | The reasoning behind a value that was not measured. |
| `copied_from` | string | with method `copied` | The profile it was copied from, as `model/method/k<K>/thinking-<mode>/category`. |

### `provenance`

<!-- fields: acceptance.Provenance -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `source` | string | yes | The document the value was read from. |
| `run` | string | no | The job that measured it, such as a GitHub Actions run. |
| `benchmark` | string | no | The prompt set, such as `SPEED-Bench Qualitative`. |
| `prompts` | integer | no | How many prompts it covered. Not negative. |
| `hardware` | string | with method `measured` | The catalog chip it was measured on. A resolution on other hardware records this. |
| `engine_image` | string | with method `measured` | The engine image or release that ran. |
| `tp` | integer | no | The tensor-parallel width it ran at. Not negative. |
| `sampling` | mapping | with method `measured` | The decoding configuration, below. |
| `settings` | mapping | no | Other engine settings that bear on acceptance, by name, such as `rejection_sample_method`. |

### `provenance.sampling`

<!-- fields: acceptance.Sampling -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `temperature` | number | yes | Not negative. 0 is greedy decoding. |
| `top_p` | number | no | In (0, 1]. |
| `top_k` | integer | no | At least −1; −1 and 0 disable it. |
| `max_tokens` | integer | no | Output cap. Not negative. |

## Inline values

A workload's `acceptance`, a scenario's `acceptance.value`, and the value a command line
supplies all take this form. A value names the K it was stated for; at a deployment that
speculates another K it is refused rather than rescaled.

<!-- fields: acceptance.Value -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `num_spec_tokens` | integer | yes | The K the value was stated for. At least 1. |
| `acceptance_length` | number | one of three | AL, in [1, K + 1]. |
| `acceptance_rate` | number | one of three | (AL − 1) / K, in [0, 1]. Not with `acceptance_length`. |
| `per_position_rates` | list of numbers | one of three | K rates, as for a profile. Alone, they state AL as 1 + their sum; with either of the others, they must agree with it. |
| `source` | string | no | Where the value came from, recorded with the resolution. |

## Scenario pin

The value of a scenario's `acceptance` key.

```yaml
--8<-- "docs/contracts/examples/scenario-acceptance.yaml:acceptance"
```

To use another profile set for the run, and to accept coding's curve for traffic that
has none of its own:

```yaml
acceptance:
  profiles: inferencex-golden-al
  fallback_category: coding
```

<!-- fields: acceptance.Pin -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `profiles` | string | no | An acceptance-profile set under `acceptance/` in the registry, consulted in place of the default. Not with `value`. |
| `value` | mapping | no | An inline value, above. |
| `fallback_category` | string | no | The category to use when the workload's has no curve. The substitution is recorded. Not with `value`. |

A pin must set at least one key.

## Resolution

`acceptance.Resolve` takes the run's model, method and K, the registry sets, the
workload's terms, the scenario's pin and the command line's value, and returns the
acceptance length, the per-position rates, and which layer supplied them. The highest
layer that states acceptance wins; layers are not merged.

| Layer | Supplies acceptance by |
|---|---|
| `registry` | the profile matching (model, method, K, the workload's thinking mode and category) in the registry sets, the default |
| `workload` | the workload's own `acceptance` value |
| `scenario` | the pin's `value`, or the matching profile in the set its `profiles` names |
| `cli` | the command line's value, such as `--speculative-acceptance-rate` at the deployment's K |

--8<-- "docs/contracts/generated/enum/acceptance.Layer.md"

Nothing falls back silently. A lookup is refused when

- the workload states no thinking mode, or no category and no fallback was requested;
- no profile matches and no `fallback_category` was requested, or the fallback has no
  profile either;
- two registry sets both match;
- the scenario names a set that was not supplied. The registry is not consulted in its
  place.

A value stated for a K other than the deployment's is refused at every layer. The
resolution records the profile set and entry, the cited source, the category a fallback
replaced, the hardware a profile was measured on when it differs from the run's, and
whether the rates were derived.

## Checks

A profile set is invalid if

- `kind` is not `AcceptanceProfileSet`, `name` is empty, there are no profiles, or a
  caveat is empty;
- two profiles share a key;
- a profile has no `model` or `speculative_method`, `thinking` is not `on` or `off`, or
  `category` is malformed;
- `num_spec_tokens` is less than 1, or `acceptance_length` is not finite or lies outside
  [1, K + 1];
- `per_position_rates` has other than K entries, an entry outside [0, 1], rises, or sums
  more than 0.01 from AL − 1;
- `method` is not recognized or is `not_charged`, or a field the method requires, per the
  table above, is missing;
- a provenance has no `source`, a negative count, an empty setting, or sampling with no
  `temperature`, a negative one, a `top_p` outside (0, 1] or a `top_k` below −1.

An inline value is invalid if `num_spec_tokens` is less than 1, it states no acceptance,
it states both `acceptance_length` and `acceptance_rate`, either lies outside its range,
or its rates are invalid or disagree with it. A pin is invalid if it sets nothing, sets
both `profiles` and `value`, sets `fallback_category` with `value`, or its value is
invalid.

With a deployment in the bundle, every inline value (the scenario's pin, its trace's, its replay's, merged with its catalog entry when the bundle carries one, a catalog replay's, and
the workload shape's) must name the K of each pool that speculates. When no pool
speculates, a stated value draws a warning.

Source: [`spec/acceptance`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/acceptance).
