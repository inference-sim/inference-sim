# blis-vllm

A Go model of the [vLLM](https://github.com/vllm-project/vllm) inference engine's scheduling,
KV-cache, connector and metrics behaviour, for a discrete-event simulator. It implements the
engine-neutral contract in [blis-substrate](https://github.com/inference-sim/blis-substrate)
`contracts/` (`engine.Engine`), so a simulator such as
[inference-sim](https://github.com/inference-sim/inference-sim) builds it once and then holds only
that interface. The design is
[inference-sim#2021](https://github.com/inference-sim/inference-sim/issues/2021).

- **One release per build.** This build models vLLM **v0.32.0**
  (`5df63a7203cdb9328d824a27b682e77c7b9928be`); see `provenance.go`. Its releases are tagged
  to the vLLM release they model, `v0.32.0-blis.N` (design §14.1); the inference-sim release
  that runs each one pins it in its `go.mod`.
- **One-way dependencies.** blis-substrate ← blis-vllm ← inference-sim. blis-vllm imports
  blis-substrate's `contracts/` packages and, until the engine boundary moves (release plan §3.3,
  M2), its cost model (`kernel/`); the cost model imports nothing from this module, and
  inference-sim's simulator imports blis-vllm at one build site,
  [cmd/engine_build.go](https://github.com/inference-sim/inference-sim/blob/be78af832794aa53eb76a5cf3ba06140510e6ddd/cmd/engine_build.go).
- **Checked against vLLM itself.** CI feeds the same inputs to vLLM's own classes (Python, CPU
  only, the GPU mocked) and to this module, and compares every step. See
  [docs/testing.md](docs/testing.md) and the [feature matrix](docs/feature-matrix.md).
- **Time is a value.** Nothing here reads a clock or starts a goroutine. The simulator passes the
  simulated instant into every call.
- **Rules packs.** The per-release validation rules (`rules/`) moved here from blis-schemas. A
  caller hands them to the cost model through blis-substrate's `contracts/rules.EngineRules`, so
  the cost model imports nothing from this module.

## Layout

| Path | What |
|---|---|
| `provenance.go` | the module's identity: the vLLM release it models and its own revision |
| `rules/` | rules packs per vLLM release, and the vLLM engine block's configuration (`rules/engineconfig`) |
| `core/` | the vLLM v0.32.0 engine, diff-tested against vLLM's own classes (`core/vllm`: behind blis-substrate's `engine.Engine`, built by the `vllm` Factory) |
| `vllmengine/` | how a simulator gets the v0.32.0 engine: it registers the Factory, prices steps with the latency kernel and sizes the KV cache |
| `compat/` | the compatibility engine: inference-sim's engine as it ran before the extraction (`compat/engine`, `compat/kv`, `compat/lora`, `compat/kernelmodel`, moved verbatim, each file stating its source commit), which keeps the existing 0.29.0/0.31.0 scenarios byte-identical until they are re-baselined to 0.32.0 (inference-sim#2027), plus the leaf packages it uses and `compat/build`, the engine-side configuration the simulator used to decode itself (the scenario's KV offload block and the `--kv-offload-config` file; inference-sim#2021 E3) |
| `kerneltests/` | the latency kernel's tests that open a kernel from a vLLM engine block, its commands and docs generator, moved here from blis-latency-kernel (`kerneltests/_inpackage` runs inside blis-substrate's `kernel` package: `kerneltests/scripts/run-inpackage.sh`). The catalog and registry they read are blis-substrate's, at the release `go.mod` pins |
| `testdata/` | inputs of `compat/kernelmodel`'s tests: inference-sim's scenarios (`testdata/README.md`); the catalog and registry are the pinned blis-substrate's, through its loader |
| `docs/feature-matrix.md` | every feature our sources use, its v0.32.0 citation, owner, oracle tier and status |
| `docs/testing.md` | the oracle tiers, the shared GPU mock, provenance, golden-trace policy |

## The field list (fields.json)

`fields.json` lists every configuration field and environment variable vLLM v0.32.0 reads, each
with a status, its evidence and an estimation rule (design VLM-3, §11.2). The type is
blis-substrate's `contracts/spec/fields`; accuracy-dashboard and blis-config-search read the list
through `inference-sim fields`, and hold no list of their own (COV-3, CFG-3).

- **Names** come from vLLM's own source at the pinned commit, not by hand: `fields/gen_fields.py`
  reads the `EngineArgs`, `AsyncEngineArgs` and front-end flag fields (expanding each config
  object to `<field>.<subfield>`), every key of `vllm/envs.py`'s `environment_variables`, and
  every environment variable vLLM reads by a literal name: `os.environ.get`, `os.getenv`,
  `os.environ[...]`, `in os.environ`, `setdefault`, `envs.is_set` and `envs.X` under `vllm/`,
  `getenv` in `csrc/`, and `env::var` in the Rust front end under `rust/`, with each read's
  file and line (`--env-reads`). It also lists the NCCL, NVSHMEM and UCX variables vLLM sets for
  those libraries. The generator's docstring states the read rule and the paths it skips, each
  a separate command or a build step (`vllm collect-env`, `vllm bench`, `vllm chat`, tests,
  build scripts); a module named for another platform is not skipped. A read whose name is
  not a literal is listed in `status.json`'s `dynamic_reads`, with the reason. CI runs
  `python3 -I fields/gen_fields.py vllm-source fields/status.json --check fields.json`, which
  fails while a name vLLM reads has no entry, an entry names a field vLLM does not have, or a
  dynamic read is unlisted.
- **The Rust front end is scanned** because `VLLM_USE_RUST_FRONTEND=1` selects it on any platform,
  CUDA included (`vllm/entrypoints/cli/serve.py`, `vllm/envs.py`), and it runs in the serving
  process. Eighteen variables are read there; the ones read nowhere else are `RUST_LOG`,
  `TOKIO_WORKER_THREADS`, the `VLLM_RS_*` thread and tokenizer settings, and the Hugging Face
  cache, offline and token variables, which also change what a run loads.
- **Status, evidence and estimation** are in `fields/status.json`, edited by hand. A field is
  `modelled` only with VLM-3's evidence: an oracle trace that sets it and that the port
  reproduces, or a test that sets it and checks the result against vLLM's own. A field the port
  does not read, or reads without such evidence, is `not_modelled` and cites an open issue
  (X28). The first version was seeded from accuracy-dashboard's `vllm_arg_policy.yaml`, as
  release plan §3.3 says: "Fields that table marked inert are entered as 'inert, unproven'."
  Estimation rules start from accuracy-dashboard discussion #33. `leave_unset` means the engine
  resolves the upstream's default (X19), so a rule is `leave_unset` only where blis-vllm resolves
  it for every value: `max_num_seqs` and `max_num_batched_tokens`, by `ResolveBatch` in
  `core/vllm/batchdefaults.go`. `block_size` is `none`: the engine resolves an unstated one only
  for the models, GPUs and KV dtypes in `oracle/blocksize/TABLE.tsv` (`ResolveBlockSize` in
  `core/vllm/blocksize.go`) and refuses the rest, citing blis-vllm#68, and a field has one
  estimation rule, which cannot say that.
- **`inert_unproven` is a transition status**, allowed only until M2 (release plan §3.3). VLM-3's
  other checks are M2 work and are not here yet: the inertness proofs that turn each
  `inert_unproven` field into `proven_inert` or reclassify it, the perturbation test over every
  `modelled` field, and the value classes each modelled field's scenarios cover.
- `TestTheFieldListLoadsAndNamesThisRelease` loads the file with the substrate's strict loader,
  checks it names this release, is in canonical form, and that every cited trace exists.

## Licence

Apache-2.0. Files translated from vLLM keep vLLM's licence and name the upstream file and commit
they were derived from; see [NOTICE](NOTICE).
