# Running an SGLang scenario

BLIS runs SGLang v0.5.23 (commit 297e54fd) as an engine kind beside vLLM. A scenario selects it with
`engine: {kind: sglang, version: "0.5.23"}`, and the pool's `engine:` block states SGLang's own
`ServerArgs` field names (`mem_fraction_static`, `chunked_prefill_size`, `attention_backend`, ...). A name
that is not a `ServerArgs` field is refused naming it. The tensor- and data-parallel widths are the
pool's `parallel:` block's; an engine block that states `tp_size` must agree with it.

Two scenarios run in CI (`cmd/sglang_engine_test.go`):

- `testdata/kernel/sglang/glm-5-h200-fp8-sglang-tp8.yaml`
- `testdata/kernel/sglang/deepseek-r1-h200-fp8-sglang-tp8.yaml`

```bash
blis run --scenario deepseek-r1-h200-fp8-sglang-tp8.yaml --scenarios testdata/kernel/sglang \
  --rate 5 --num-requests 20
blis fields --scenario deepseek-r1-h200-fp8-sglang-tp8.yaml --scenarios testdata/kernel/sglang
```

`blis fields` prints the engine's field list: every `ServerArgs` field and environment variable of the
release with its status (`modelled`, `proven_inert`, `inert_unproven`, `not_modelled`). A `not_modelled`
field set away from its default is refused by name, citing the issue that will model it.

## What a scenario must state

- `context_length`: the catalog graphs of most models carry no context window, so the host refuses a
  scenario that omits it. Use the checkpoint's `max_position_embeddings`.
- The chip is the H200: the memory CUDA holds before a model loads is measured there only.
- No speculative decoding, DP attention, pipeline parallelism, HiCache or P/D disaggregation: each is
  refused naming its open issue.

## What decides a step

SGLang's own scheduler port, KV pool sizing and CUDA-graph padding decide every step; the latency kernel
prices the pass with SGLang's engine rules and lowering (blis-sglang `rules/`, `lower/`). One named
substitution remains: the host-side facts (model, hardware, parallelism, KV store, host overheads) are read
through blis-vllm's adapter from a copy of the scenario in vLLM form, so a host-side setting that copy
resolves by vLLM's rules is vLLM's, not SGLang's. See `adoptSGLangTwin` in `cmd/sglang_engine.go`.
