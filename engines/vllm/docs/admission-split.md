# Admission: the queue's shape in the plan

Contents: [Logic](#logic) | [The assumption about the 40 ms](#the-assumption-about-the-40-ms) | [Inputs](#inputs) | [Reproduce](#reproduce) | [Cost](#cost) | [Limits](#limits)

## Logic

A request waits in a queue before the engine admits it. Whether requests contend there is a
property of the engine's own settings, so the engine states it in its plan, and the simulator
queues requests on that shape without knowing what the servers are.

`Engine.Plan().Host.Admission` (the v0.32.0 engine in `core/vllm`; the compatibility engine for older releases does the same in `compat/contract`) is a `plan.AdmissionQueue`: `Servers` independent first-in
first-out queues of `WorkersPerServer` servers each. This engine fills it from two settings:

- `renderer_num_workers` is `WorkersPerServer`. vLLM v0.32.0 runs the tokenizer and the chat
  template on the renderer's thread pool, `ThreadPoolExecutor(max_workers=renderer_num_workers)`,
  default 1 (`vllm/renderers/base.py:139-158`, `vllm/renderers/hf.py:965`,
  `vllm/config/model.py:408`).
- `api_server_count` is `Servers`. Each API-server process has its own pool, and the count's
  default depends on the load-balancing mode (`vllm/entrypoints/cli/serve.py:107-137`). An
  unstated count is the data-parallel size for a group behind the internal balancer, else 1; no
  render threads is no queue.

Until blis-schemas carries the two settings, the host gives them with `SetFrontend` (the
`vllm.Frontender` interface; unset, vLLM's defaults: one thread, the per-mode server count).

What a request asks of that queue (`Service`, the time it holds a server, and `Delay`, the
time it only waits) is the cost model's answer, `Environment.HostDemand`, from the kernel's
`AdmissionDemand`. This engine prices nothing here.

Under vLLM's Rust front end `api_server_count` is forced to 1 and the front end is
multithreaded (`serve.py:107-137`, `serve.py:274-280`). The plan does not represent that case
separately: it states the same queue shape, and the kernel supplies the Rust per-token rate as
`Service`.

## The assumption about the 40 ms

The per-request term, 40 ms (`host_admission_per_request`, method `assumed`), is a latency of
unknown origin: the registry's rationale says the tokenizer, template, message encoding and
hop measured 0.0153 and 0.1083 ms in all. The citations above show that tokenizer and template
work uses the thread pool. They do not show that the 40 ms is a delay after the pool. The split
is therefore an assumption, `Delay` holds no server, with two bounds on what it costs if wrong:

- All of the 40 ms is server work (the model before this change): one API server starts at
  most 24.8 requests a second, and a closed loop of many clients sits at that rate.
- None of it is server work (`Service` only the per-token term, as here): the cap is the
  tokenizer's own rate, about 2700 requests a second at 1k tokens.

The truth lies between if part of the 40 ms is work on the pool. On the 60 development points of
P/D run B the bounds are TTFT MAPE 78.1% (all serial) and 24.7% (this change); they bound the
model, they do not validate either reading.

## Inputs

The substrate commit that adds `plan.AdmissionQueue` (`HostPlan.Admission`) to
`contracts/engine/plan` (branch `m2/18-admission-split`). This module's `go.mod` pin moves to
that commit when the substrate layer is published.

## Reproduce

```
go test ./core/vllm -run AdmissionQueue
go test ./compat/contract -run AdmissionQueue
```

Expected: pass. With `Plan` not setting `Host.Admission` both fail.

## Cost

Constant time and memory.

## Limits

`renderer_num_workers` and `api_server_count` stay `not_modelled` in `fields.json`: `modelled`
needs a recorded vLLM trace that sets the field, and the oracle's boundary replaces the
tokenizer, so none exists. The maintainer decides what evidence would change the status. The
simulator's use of the shape is development-split evidence only (see inference-sim's page).
