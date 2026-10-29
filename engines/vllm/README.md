# blis-vllm

A Go model of the [vLLM](https://github.com/vllm-project/vllm) inference engine's scheduling,
KV-cache, connector and metrics behaviour, for a discrete-event simulator. It implements the
engine-neutral contract in [blis-schemas](https://github.com/inference-sim/blis-schemas)
(`engine.Engine`), so a simulator such as
[inference-sim](https://github.com/inference-sim/inference-sim) builds it once and then holds only
that interface. The design is
[inference-sim#2021](https://github.com/inference-sim/inference-sim/issues/2021).

- **One release per build.** This build models vLLM **v0.32.0**
  (`5df63a7203cdb9328d824a27b682e77c7b9928be`); see `provenance.go`. blis-vllm's first release is
  **v0.32.0**, tagged to the vLLM release it models; it is what
  [inference-sim v0.14.0](https://github.com/inference-sim/inference-sim/releases/tag/v0.14.0) runs.
- **One-way dependencies.** blis-schemas ← blis-latency-kernel ← blis-vllm ← inference-sim.
  blis-vllm imports blis-schemas and blis-latency-kernel (v0.8.0 and v0.4.0 in `go.mod`); the
  kernel imports only blis-schemas, and inference-sim's simulator imports blis-vllm at one build
  site, `cmd/engine_build.go`.
- **Checked against vLLM itself.** CI feeds the same inputs to vLLM's own classes (Python, CPU
  only, the GPU mocked) and to this module, and compares every step. See
  [docs/testing.md](docs/testing.md) and the [feature matrix](docs/feature-matrix.md).
- **Time is a value.** Nothing here reads a clock or starts a goroutine. The simulator passes the
  simulated instant into every call.
- **Rules packs.** The per-release validation rules (`rules/`) moved here from blis-schemas. A
  caller hands them to the latency kernel through blis-schemas' `rules.EngineRules`, so the kernel
  imports nothing from this module.

## Layout

| Path | What |
|---|---|
| `provenance.go` | the module's identity: the vLLM release it models and its own revision |
| `rules/` | rules packs per vLLM release, and the vLLM engine block's configuration (`rules/engineconfig`) |
| `core/` | the vLLM v0.32.0 engine, diff-tested against vLLM's own classes (`core/vllm`: behind blis-schemas `engine.Engine`, built by the `vllm` Factory) |
| `vllmengine/` | how a simulator gets the v0.32.0 engine: it registers the Factory, prices steps with the latency kernel and sizes the KV cache |
| `compat/` | the compatibility engine: inference-sim's engine as it ran before the extraction (`compat/engine`, `compat/kv`, `compat/lora`, `compat/kernelmodel`, moved verbatim, each file stating its source commit), which keeps the existing 0.29.0/0.31.0 scenarios byte-identical until they are re-baselined to 0.32.0 (inference-sim#2027), plus the leaf packages it uses and `compat/build`, the engine-side configuration the simulator used to decode itself (the scenario's KV offload block and the `--kv-offload-config` file; inference-sim#2021 E3) |
| `kerneltests/` | the latency kernel's tests that open a kernel from a vLLM engine block, its commands and docs generator, moved here from blis-latency-kernel (`kerneltests/_inpackage` runs inside the kernel's package: `kerneltests/scripts/run-inpackage.sh`) |
| `testdata/` | inputs of `compat/kernelmodel`'s tests: copies of inference-sim's vendored catalog, registry and scenarios (`testdata/README.md`) |
| `docs/feature-matrix.md` | every feature our sources use, its v0.32.0 citation, owner, oracle tier and status |
| `docs/testing.md` | the oracle tiers, the shared GPU mock, provenance, golden-trace policy |

## Licence

Apache-2.0. Files translated from vLLM keep vLLM's licence and name the upstream file and commit
they were derived from; see [NOTICE](NOTICE).
