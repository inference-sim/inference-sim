# Decoder replay: the rows the later layers run

DeepSeek-V4.1 at vLLM v0.32.0 does not run all forty layers on every token of a prefill step. This page says which layers run on which rows, how the engine states that to the latency kernel, how to reproduce the figures, and what is not priced.

Contents: [What vLLM does](#what-vllm-does) · [What the cost model is told](#what-the-cost-model-is-told) · [Inputs](#inputs) · [Reproduce](#reproduce) · [A worked example](#a-worked-example) · [Cost](#cost) · [What this does not show](#what-this-does-not-show) · [Sources](#sources)

## What vLLM does

DeepSeek-V4.1-Flash has 40 layers. Four of them produce the KV that the other layers read (`kv_source_layer_ids` is `[2, 8, 14, 20]`); the last is layer 20. The layers after it own only sliding-window KV, a window of 128 tokens.

With the V2 model runner and `swa_bounded_replay` on (its default), vLLM v0.32.0 runs layer 20 and layers 21 to 39 on a sub-batch: the last `min(scheduled, 128)` tokens of each request in the step. Layers 0 to 19, the embedding and the drafter keep every row. Layer 20 first writes its KV for every row (its input projections, cache insert and compressor), then reruns on the sub-batch. The step still launches every layer. The code is `vllm/models/deepseek_v41/nvidia/model.py:800-835,1160-1245`, `decoder_replay_layers.py` and `nvidia/model_state.py:438-540` at v0.32.0 (`5df63a7203cdb9328d824a27b682e77c7b9928be`). The feature is not in v0.31.0.

Which rows. A request keeps its last 128 tokens of the tokens it schedules *this step*, not of its prompt: every chunk of a chunked prefill keeps its own last 128. A request with a prefix-cache hit schedules the hit's 128-token rewind plus the tokens after it, and keeps the last 128 of those. A decode or verify row schedules at most 1 + k tokens and keeps all of them, so a decode step is unchanged.

Which steps. The sub-batch exists in steps that run with no CUDA graph, and in piecewise-graph steps padded to `decoder_replay_trim_threshold` (768) tokens or more. A full graph, and a piecewise graph below 768, keep the whole batch. The model runner picks no graph when the step has more tokens than the largest captured size (`max_cudagraph_capture_size`, default 512), which is every prefill chunk of an 8192-token budget.

What `decoder_replay_cudagraph_capture_sizes` does. The sub-batch itself runs in a piecewise graph of its own when `cudagraph_decoder_replay` is on (its default; the engine block has no key to turn it off) and the sub-batch fits a captured size: it is padded up to the smallest size that holds it. Unstated, the sizes are multiples of 128 up to `min(max_cudagraph_capture_size, max_num_seqs × 128)`, coarser above 2048. A larger sub-batch runs eager at its own size. A stated size above `max_num_batched_tokens` is a startup error.

What turns it off (`_decoder_replay_supported`, `model.py:1202-1238`): the model runner is V1; prefill context parallelism; the model's own sequence parallelism; micro-batching; the pipeline stage holding layer 20 not holding every layer after it; an Engram layer at or after layer 20; a drafter whose window is wider than 128 or that has a layer that is not sliding-window attention.

## What the cost model is told

The latency kernel prices every layer on every scheduled token unless the batch says otherwise. The engine says so in neutral words (blis-substrate `kernel.Batch.TailFrom`, `ReqShape.TailRows`, `TailPadTokens`; in a step shape `StepShape.TailFrom`, `RequestStep.TailRows`, `PaddedShape.TailTokens`): the first layer that runs on fewer rows, each request's rows, and the rows its batch is padded with. The kernel prices the layers before that layer on the step's shape and the rest on a shape in which each request schedules its last rows, with the earlier ones counted as computed context.

`compat/kernelmodel/decoder_replay.go` states them from vLLM's rules above:

- The pool has the mechanism when the model is DeepSeek-V4.1, the release is 0.32.0 and `Model.PrefixHit` resolves bounded replay on (the same resolution that cuts a prefix hit's rewind), and none of the conditions in the last paragraph above holds. The first layer is the last layer in the model graph whose kind owns a compressed KV cache; the window is the graph's window cache.
- A step trims when the kernel's graph resolution runs it with no graph (a mode with none for it, or more tokens than the largest captured size), or in a piecewise graph whose padded size, the smallest captured size that holds the step, is 768 or more.
- Each request keeps `min(scheduled, window)` rows; the sub-batch is padded to the smallest of the replay graph sizes that holds all the kept rows when the pool runs replay graphs (the kernel's mode for non-decode steps has piecewise graphs), else it runs at its own size.
- The assumption is disclosed in the engine's assumptions: `decoder_replay = on: ...`, or `decoder_replay = off: <reason>`.

A data-parallel pool is priced as before (every layer on every token), and says so, because vLLM agrees the sub-batch across ranks and this model does not. A pipeline-parallel V4.1 pool is refused by the kernel for another reason.

## Inputs

| Input | Where it comes from |
|---|---|
| Behaviour | vLLM v0.32.0, `5df63a7203cdb9328d824a27b682e77c7b9928be`, the files cited above |
| Model graph | blis-substrate's catalog, `deepseek-v4.1-flash` (layer kinds, caches, window); the cut is read from the graph, not from `config.json` |
| Capture sizes and graph modes | the kernel's resolution of the pool's `cudagraph_mode`, `max_cudagraph_capture_size` and `cudagraph_capture_sizes` (`Kernel.CaptureTokens`, `CaptureSizes`, `GraphModes`) |
| Replay graph sizes | `compilation_config.decoder_replay_cudagraph_capture_sizes`, or the default above |
| Trim threshold | 768, `CompilationConfig.decoder_replay_trim_threshold`'s default; the engine block has no key for it |
| Cost | blis-substrate's kernel, the commit this repository pins |

## Reproduce

```
go test ./compat/kernelmodel -run 'TestAPrefillChunkIsPriced|TestTheSavingGrows|TestACachedPrefix|TestADecodeStepAndASmall|TestAPiecewiseStepTrims|TestTheStatedReplayGraphSizes|TestAPoolThatCannotReplay|TestPricingAStep' -count=1 -v
```

The tests open DeepSeek-V4.1-Flash on one 8-GPU H200 node at tensor parallel 8 under release 0.32.0 (`replayScenario` in `decoder_replay_test.go`) and price steps through `Model.StepTimeOfShapes`. The same pool with `swa_bounded_replay: false` is the price of every layer on every token.

## A worked example

H200, tensor parallel 8, DSpark with five drafts, `block_size` 64, `max_num_batched_tokens` 8192, the default capture sizes. Step prices in microseconds, with the decoder replay and without it (`swa_bounded_replay: false`, which prices every layer on every token):

| Step | With | Without |
|---|---|---|
| one chunk of 2048 tokens, no context | 161,090 | 199,842 |
| one chunk of 8192 tokens, no context | 296,521 | 459,529 |
| one chunk of 8192 tokens after 300,000 cached | 344,440 | 655,699 |
| two chunks of 4096 | 298,842 | 457,113 |
| sixteen decodes of six tokens | 22,876 | 22,876 |
| one chunk of 300 tokens (a piecewise graph below 768) | 32,581 | 32,581 |

The saving grows with the rows the later layers do not run, a little more than four times from 2048 to 8192 tokens, and with the context the skipped rows would have scored. Two chunks keep twice the rows of one and save a little less. Decode steps and small graph steps are not changed.

## Cost

Not benchmarked. A step that trims clones its batch once, and the kernel then prices a second shape through the same per-layer pricing, so pricing such a step costs up to about one more pricing of the step; a step that does not trim costs what it did.

## What this does not show

- Layer 20 is priced wholly on the kept rows. The rows outside the window also pay layer 20's KV write in vLLM. In the kernel's graph that work (input projections, cache insert, compressor, indexer key) is 1.1 ms of a 157 ms step at 2048 tokens and 1.8 ms of 288 ms at 8192, 0.6% to 0.7%.
- The replay graph's launch saving. A sub-batch that replays a graph costs less host time to launch than an eager one; the kernel launches the same kernels either way and the step's launch term is unchanged.
- A request that asks for prompt logprobs keeps every row in vLLM; no BLIS request does.
- Data parallelism, where the sub-batch is agreed across ranks (priced as before).
- The drafter. A DSpark drafter's context stack keeps its pass over every scheduled token.
- Whether the size of the effect is right on hardware. The figures above come from the kernel's coefficients; the check against a recorded run is the TTFT comparison in the pull request that adds this page, which is in-sample on the development split of the sixteen H200 rows.
- Oracle coverage. The rows are a pure function of the scheduler's per-request scheduled tokens, which the scheduler oracle already compares, and of `model_state.py:466-470`, a worker-side line that needs a GPU; no oracle compares it. See the pull request's open list.

## Sources

- vLLM v0.32.0 (`5df63a72`): `vllm/models/deepseek_v41/nvidia/model.py`, `nvidia/model_state.py`, `nvidia/decoder_replay_cudagraph.py`, `decoder_replay_layers.py`, `vllm/config/compilation.py:559-575`, `vllm/v1/worker/gpu/cudagraph_utils.py:518-548`, `vllm/v1/core/sched/scheduler.py:3105-3120`. Apache-2.0.
- blis-substrate: `kernel/tail.go`, and the section "Layers that run on some of the rows" of its kernel step concepts page.
