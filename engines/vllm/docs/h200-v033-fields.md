# The three settings of the DeepSeek-V4.1-Flash H200 rows

Contents: [Why this page](#why-this-page) | [PYTHONUNBUFFERED](#pythonunbuffered) | [Synthetic rejection sampling](#synthetic-rejection-sampling) | [The Rust front end](#the-rust-front-end) | [How a harness states them](#how-a-harness-states-them) | [Reproduce](#reproduce) | [Limits](#limits)

## Why this page

The sixteen InferenceX rows of DeepSeek-V4.1-Flash on H200 (vLLM v0.33.0, agentic traces, DSpark
speculation with five drafts) were refused by the published scorer for three reasons, one per
setting: `env.PYTHONUNBUFFERED` was not in the field list, and `env.VLLM_USE_RUST_FRONTEND` and
`speculative_config.rejection_sample_method` were `not_modelled` with the rule `none`. This page says
what each setting does in vLLM v0.32.0 (the release this engine models; v0.33.0 is noted where it was
read), how the engine now treats it, and what stays open. The list is `fields.json`; the engine block
that states the settings is described in [engine-config.md](engine-config.md).

## PYTHONUNBUFFERED

**What it does.** CPython reads it at start-up and writes standard output and standard error without
a buffer. A log line reaches the pipe when it is written instead of when a buffer fills.

**What vLLM does with it.** Nothing. No file under `vllm/`, `csrc/` or `rust/` names it at v0.32.0 or
at v0.33.0 (the scan of `fields/gen_fields.py` read 4120 files at v0.32.0; `git grep` over the same
directories at v0.33.0 finds none). It appears only in CI files and a deployment guide.

**Classification.** `proven_inert`, with a proof of kind `unread`
(`oracle/inertness/PYTHONUNBUFFERED.json`), the same argument as `PYTHONNOUSERSITE`. It is a reviewed
static fact, not a run: that buffering changes when bytes are delivered and no engine decision is a
reading of the interpreter's documented behaviour. It joins the list through `extra_env` because no
file of the serving process names it.

**In the engine.** An engine block that states it is accepted (`inertEnv` in `core/vllm/config.go`) and
runs step for step as one that does not. Any other variable the engine does not read is still refused
by name. A harness records the variable on the point and does not write it to the scenario.

## Synthetic rejection sampling

**What vLLM does.** With `rejection_sample_method: synthetic` the rejection sampler does not compare
draft tokens with the target. It accepts position `i` with a stated probability. The run supplies the
mean acceptance length `L` as `synthetic_acceptance_length`, and vLLM turns it into per-position
unconditional rates by the minimum-variance schedule (`vllm/config/speculative.py:556-564`): with
`n` drafts, `floor(L-1)` positions have rate 1, the next has the fractional part of `L-1`, and the
rest have 0. The sampler converts them to conditional rates (`unconditional_to_conditional_rates`,
`vllm/v1/spec_decode/utils.py:848`) and accepts draft `pos` when a uniform draw is below that rate
and the draft is not a `-1` padding token (`vllm/v1/sample/rejection_sampler.py:759-763`). The
setting is refused without a length or rates (`speculative.py:575`), a length outside `[1, n+1]` is
refused (`:595`), and a length with any other method is refused (`:1841-1846`).

The rows state `L = 3.51` with `n = 5`: rates `[1, 1, 0.51, 0, 0]`. Every step accepts two drafts, and
a third with probability 0.51. The server log's realised mean is 3.511.

**What the engine does.** It accepts the setting when the run supplies a length
(`speculative_config.synthetic_acceptance_length`), as the run's supplied acceptance; the length is
the pin the acceptance resolver reads, at the rank of the scenario's own pin. It refuses `synthetic`
without a length by name, as vLLM does. `block` and `standard` are accepted because the acceptance is
supplied by the run and the engine decides nothing else from the method.

**Does the model match?** In the long-run mean and in the rates, yes; in the per-step shape, not
exactly.

- The rates are vLLM's own: `oracle/synthetic_acceptance` runs `_resolve_synthetic_acceptance_rates`
  over a grid of lengths and drafts, and `TestSyntheticAcceptanceLengthIsVLLMsOwnResolution` holds
  blis-substrate's `acceptance.MinimumVarianceRates` to it bit for bit.
- The Triton sampler that consumes them is not compared: it needs a GPU.
- The simulator does not draw from the rates. It accrues `alpha x drafts` each step with `alpha =
  (L-1)/n` and accepts the whole part (`sim/engine_driver.go`, `outcome`). With all five drafts
  verified this gives two or three accepted drafts a step with the mean 2.51, the support and the mean
  of the minimum-variance schedule, deterministically instead of by a Bernoulli draw on the third
  position. The variance is lower, which moves no mean latency.
- The two differ when fewer than `n` drafts are verified in a step (a request near its output limit,
  or a step trimmed by adaptive verification). vLLM accepts by position, so two verified drafts are
  both accepted; the simulator accepts `alpha x 2`. The engine caps a request's accepted drafts at
  those it verified, and rejects `-1` padding drafts, as the sampler does. The rows' realised mean
  (3.511 against 3.51) suggests trimming did not bind.

**Classification.** `speculative_config.rejection_sample_method` is `modelled`. Its classes are
covered by the oracle record, by a pricing test on this model and chip, and by the engine block's
tests of `standard` and `block`; the Triton sampler is not among the comparisons.

## The Rust front end

**What vLLM does.** `VLLM_USE_RUST_FRONTEND=1` makes `vllm serve` start the `vllm-rs` binary in place of
the Python API server (`vllm/entrypoints/cli/serve.py:65-66, 346-355`). Reading the source:

- `api_server_count` is 1. Unstated, it defaults to 1 (`serve.py:109, 114`); stated above 1 it is
  reduced to 1 with a warning (`:131-135`), and `run_multi_api_server` refuses a count above 1
  (`:277-280`).
- The front end is one process with a multi-threaded Tokio runtime, worker threads the CPU count
  capped at 32 (`rust/src/cmd/src/main.rs:22-45`). Each chat request is rendered and tokenized in
  `spawn_blocking` (`rust/src/chat/src/lib.rs:184`), which has its own thread pool, and the arrival
  time is stamped before rendering so the render counts toward time to first token (`:182`). The
  Python renderer's `renderer_num_workers` thread pool is not used.
- The tokenizer is loaded through fastokens, which the registry prices at one tenth of the Python
  path's per-token rate.
- It forces `mm_device_do_normalize` off (`vllm/config/model.py:1075-1096`), which matters only for
  multimodal input. The scheduler, the KV cache and the model runner do not read the variable.

**What the engine does today.** `env: {VLLM_USE_RUST_FRONTEND: "1"}` is read into the engine block's
`frontend` and reaches the latency kernel, which charges admission at the registry's
`host_admission_per_token_rust_frontend` instead of `host_admission_per_token`. On this model and chip
a 290,000-token prompt (the rows' mean input at one client) is admitted in 77 ms instead of 774 ms.
The Rust rate is the registry's, method `assumed`: a vendor speedup (fastokens' stated factor of ten)
applied to the Python-path rate, not a measurement. It is recorded as an assumption on the point. Where
the registry lacks the entry the kernel keeps the Python rate and says so.

**Classification.** `modelled`: the engine reads the field and it changes a decision, and tests state
it (`TestTheRustFrontendIsPricedAtItsAdmissionRate`, the scenario
`vllmengine/testdata/deepseek-v41-flash-h200-tp8-dspark5-synthetic-rust.yaml`). The evidence is
pricing tests and no recorded vLLM trace: the oracle's boundary replaces the tokenizer, so there is
none to compare. A harness must not read `modelled` here as a measured rate.

**How the admission split should treat it.** The admission queue of the host-service contract is
`Servers` queues of `WorkersPerServer` servers. For a pool on the Rust front end it should state
`Servers = 1`, whatever `api_server_count` says (vLLM forces it), and `WorkersPerServer = 0` (no render
queue: the blocking pool is far larger than the rows' 128 clients, and `renderer_num_workers` is not
read), so the request waits only its delay. The kernel's Rust per-token rate is then the delay, and
no queue builds. That is a refusal-free model; its missing piece is a measured Rust rate and a bound
on concurrent renders per core, either of which would let the queue be stated with a finite server
count. Until then a run that saturates the front end cannot be told from one that does not.

## How a harness states them

- `env.PYTHONUNBUFFERED`: `proven_inert`, recorded on the point, not written.
- `env.VLLM_USE_RUST_FRONTEND`: `modelled`, written as the engine block's
  `env: {VLLM_USE_RUST_FRONTEND: "1"}` (the engine refuses the `frontend` key and asks for this form).
- `speculative_config.rejection_sample_method`: `modelled`, written with
  `speculative_config.synthetic_acceptance_length` in the same block when the method is `synthetic`.

## Reproduce

```
python3 -I fields/gen_fields.py <vllm v0.32.0> fields/status.json --value-classes fields/value_classes.json --check fields.json
python3 -I fields/gen_classes.py <vllm v0.32.0> fields.json --check fields/value_classes.json
python3 -I oracle/inertness.py <vllm v0.32.0> oracle/inertness/PYTHONUNBUFFERED.json --check
go test . -run TestTheH200Rows
go test ./core/vllm -run TestAnInertEnv
go test ./compat/kernelmodel -run 'TestTheRustFrontend|TestSyntheticRejectionSampling|TestAContainerVariable'
go test ./vllmengine -run TestTheH200Recipe
```

Cost: the generators take about a minute each; the tests a few seconds.

## Limits

- The Rust admission rate is an assumption, not a measurement.
- The simulator's per-step acceptance is deterministic; it equals vLLM's schedule only when every
  draft is verified.
- The recorded rows ran v0.33.0 and the engine models v0.32.0. Between the two tags `serve.py` changes
  only its imports, the synthetic schedule and the sampler's accept rule are unchanged, and the Rust
  front end's `spawn_blocking` render is unchanged; v0.33.0 adds a fourth method, `fly`, which it
  refuses with synthetic sampling. Other v0.33.0 differences are not covered here.
