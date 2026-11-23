# Deferred items in M3.2

What M3.2 deliberately leaves out, why, the issue that tracks it, and what resumes it. An item here
is not forgotten and not closed: its issue stays open, and a refusal in the code cites it (X28). When
the work lands, the row moves out of this page in the same change.

## Left out by decision

| Item | Issue | Why it is not in M3.2 | Where the code refuses it |
|---|---|---|---|
| `blis replay` with the autoscaler or node pools | [#1314](https://github.com/inference-sim/inference-sim/issues/1314) | Both features are run-only in M3.2, and replay refuses them at startup rather than run without them. | `cmd/replay.go` stops at startup with a message that names the flag or the policy-bundle section |
| Replaying a trace from a multi-node fleet | [#2146](https://github.com/inference-sim/inference-sim/issues/2146) | Follows from node pools being run-only: replay cannot rebuild the fleet the trace came from, so it refuses the trace rather than replay it on a different one. | `cmd/replay.go`; `sim/workload/fieldstatus.go` (`MaxNodesSpanned`) |
| LoRA placement policies | [#1493](https://github.com/inference-sim/inference-sim/issues/1493), [#1495](https://github.com/inference-sim/inference-sim/issues/1495) | Outside the scope of M3.2. | none |
| Refitting the `cost-model-memory` coefficient family (CUDA-graph capture, the non-torch budget, peak activation) | none; the visible effect is in [#2521](https://github.com/inference-sim/inference-sim/issues/2521) | Ruled out of every M3.2 refit round. The family stays at its published values. | none |
| TensorRT-LLM as an engine | none | Not in scope. The engines are `engines/vllm` and `engines/sglang`. | none |
| Removing the `emit` markings and the `tp` layouts from the catalog graphs and the deriver | none | Doing so changes the catalog content hash recorded in six byte-gate goldens, and 23 graphs cannot be re-derived offline. Only the `emit`-conditions step landed. | none |

Resume a "left out by decision" item when the scope is reopened: write the design first, then the
change, as for any new module behaviour.

## Held back by a gate or by a dependency

| Item | Issue | Why | What is needed to resume |
|---|---|---|---|
| Refit of `host_launch_per_chip`, `host_admission_per_request` and `host_output_token` | none | Each failed its pre-registered plan gate; the negative results are recorded with the refit. Only the DeepSeek-V4-Pro H200 launch coefficient passed ([#2510](https://github.com/inference-sim/inference-sim/issues/2510)). A forced fit would overfit, so none was made. | New measurements with a published fit role |
| The official held-out gate, the ledger release job, real-seed roles and the partition write | none | They run after the merge, from a release; they are not a precondition of any M3.2 change. | The merge, then a release |
| `ported_functions_unpublished` for points at another engine commit, in the nightly run | none | The nightly run passes neither `--engine-report` nor `--engine-checkout`, so the check has no report to read. `engines/upgrade/upgrade_report.py` produces the report; nothing calls it for the nightly run. | A step that runs the report producer and passes its output |
| SGLang `disable_priority_preemption=false` | [#2451](https://github.com/inference-sim/inference-sim/issues/2451) | No oracle scenario covers the class, and the mock-list citation text for it was reverted. Covering it needs a new oracle recording. | A SGLang version bump with a provenance refresh of the goldens |
| KV capacity when `gpu_memory_utilization` is unstated: vLLM's extensible KV-cache default, and the demand-side bytes per token | [#2521](https://github.com/inference-sim/inference-sim/issues/2521), [#2330](https://github.com/inference-sim/inference-sim/issues/2330), [#2290](https://github.com/inference-sim/inference-sim/issues/2290), [#2316](https://github.com/inference-sim/inference-sim/issues/2316) | The source audit is done. Modelling the default moves the capacity of every scenario that leaves utilization unstated by whole GiB, which contradicts the committed oracle recordings (taken at vLLM v0.33.0); the rest depends on registry coefficients that are not refit. | A decision on the recordings, with demand-side KV accounting, gated by the oracle |
| `cpu_offload_gb` (vLLM weight offload) and `default_chat_template_kwargs.thinking_mode` on agentic points | [#2622](https://github.com/inference-sim/inference-sim/issues/2622), [#2482](https://github.com/inference-sim/inference-sim/issues/2482) | `cpu_offload_gb` needs a PCIe term in the cost model and an effect on the KV budget. `thinking_mode` is a front-end setting and needs a new kind of proof. Neither alone unlocks the points that use both. | A decision on scope for each |

## Planned follow-ups

These are started or scheduled, not abandoned.

- [#2438](https://github.com/inference-sim/inference-sim/issues/2438): the fused-kernel launch count that graph capture still counts, after the final comparison.
- [#1924](https://github.com/inference-sim/inference-sim/issues/1924): the CPU KV-offload tier ignoring model-resident host memory.
- [#2349](https://github.com/inference-sim/inference-sim/issues/2349): refusal tests for every refused SGLang surface item.
- [#2350](https://github.com/inference-sim/inference-sim/issues/2350): inertness proofs for SGLang fields. The harness and the first proof (`stream_interval`) exist; the remaining fields and environment variables are listed in the issue, which stays open.
