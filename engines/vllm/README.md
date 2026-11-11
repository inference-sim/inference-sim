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
- **A configuration field with no command-line flag is not listed.** The generator takes a field
  of `EngineArgs` or `AsyncEngineArgs` only if some `add_argument` call names it. Six such fields
  have none: `model_weights`, `_api_process_count`, `_api_process_rank`, `enable_mm_processor_stats`,
  `mamba_config` and `tokens_only`. vLLM's own code sets them, or they are reached through another
  flag. `enable_mm_processor_stats` is an `ObservabilityConfig` flag that its source calls "for
  internal use only (e.g., benchmarks)" and "not exposed as a CLI argument"
  (`vllm/config/observability.py:94`), so a deployment cannot set it. If a later release adds a
  flag for it, the generator lists it and the completeness check demands a status.
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
  `core/vllm/batchdefaults.go`, and `gpu_memory_utilization`, whose default 0.92 `resolveEngine`
  in `compat/kernelmodel/engine_settings.go` fills in for every release the engine models.
  `gpu_memory_utilization` stays `not_modelled`: the KV budget it sizes is priced by the cost
  model, and nothing here compares that budget with vLLM's own memory profiling (VLM-3). `block_size` is `none`: the engine resolves an unstated one only
  for the models, GPUs and KV dtypes in `oracle/blocksize/TABLE.tsv` (`ResolveBlockSize` in
  `core/vllm/blocksize.go`) and refuses the rest, citing blis-vllm#68, and a field has one
  estimation rule, which cannot say that.
  A field's estimation rule applies when the run did not record the value. It never replaces a
  value the run recorded: a stated value is carried into the scenario as stated, whatever the
  field's status. A not-modelled field that the engine reads and prices when it is stated is a
  status to correct (modelled, with its evidence, or refused when stated, VLM-3); the rule does
  not stand in for that. The v0.32.0 engine refuses a not-modelled key of the engine block that nothing reads when it is
  stated away from vLLM's default (`TestEveryNotModelledFieldIsRefusedOrExplained` walks fields.json
  and gives each refusal a partner that states the default and builds), naming the field and citing
  its issue. The keys the latency kernel reads and prices at vLLM v0.31.0's policy (`blis-vllm#73`
  to `#80`, `#83`, `#8`) are the walk's allow-list.

  Those fields stay `not_modelled` in M2, and the list is the M3 scope: the kernel accepts and prices a stated value of
  each at its vLLM v0.31.0 policy, and updating that policy to v0.32.0's is M3 (blis-vllm#78 holds the date), four
  fields at a time. `cudagraph_mode` and `cudagraph_capture_sizes` are the exception: they are `modelled`, with the
  divergence disclosed in the engine's assumptions. A point that states one of the others is priced at the
  kernel's v0.31.0 policy, and its substitution list should say so. The fields, with the kernel site that
  prices each: `all2all_backend` (`kernel.go:798`), `disable_custom_all_reduce` (`internal/resolve/layout.go:208`),
  `disable_cascade_attn` (`new.go:2868`), `gpu_memory_utilization` (`pipeline.go:573`), `mamba_cache_dtype`
  (`new.go:270`), `moe_backend` (`step_terms.go:393`), `enable_dbo` (`new.go:708`), `enable_eplb` and
  `eplb_config.num_redundant_experts` (`new.go:1353`), the three `engram_config` keys (`draft.go:432`),
  `dcp_comm_backend` (`new.go:550`), `dcp_q_replicate` (`new.go:533`), `cp_kv_cache_interleave_size`
  (`internal/resolve/dcp.go:88`) and `attention_config.use_trtllm_attention` (`settings_priced.go:203`, which prices
  `true` on one lane and refuses `false`). The sites are in blis-substrate's `kernel` package; they were read at
  the pinned layer and are not diffed against v0.32.0.

  Every `not_modelled` field with the rule `leave_unset` is named below, with what a deployment that
  states it gets, so leaving it unset gives vLLM's default (`TestReadmeNamesEveryLeaveUnsetField`
  fails when this list and fields.json differ):

  - not a key of the engine block, so a deployment that states it is refused when the block
    is decoded (an unknown key), and a run that does not state it runs vLLM's default: `attention_backend`,
    `attention_config.backend` (blis-vllm#80), `attention_config.indexer_sparse_logits` (#84),
    `speculative_config.synthetic_acceptance_length` (#88), `kv_offloading_backend` (#87) and the
    environment variable `VLLM_USE_SIMPLE_KV_OFFLOAD` (#87, refused by the block's `env`);
  - a deployment field, not a key of the engine block, that the engine reads from the deployment:
    `data_parallel_external_lb` (#86) and `data_parallel_hybrid_lb` (#64), both stated as the pool's
    `parallel.data_parallel_load_balancing`, and `data_parallel_size_local` (#55), stated as
    `parallel.dp_local` and refused when its packing differs from the engine's;
  - read by the latency kernel, which prices a stated value at vLLM v0.31.0's policy:
    `gpu_memory_utilization` (#73) and `engram_config.cpu_offload` (#78), where `leave_unset` fills
    only a value the run did not record;
  The two CUDA-graph switches `cudagraph_decoder_replay` (decoder-replay graphs,
  blis-vllm#80) and `cudagraph_specialize_lora` (blis-vllm#78) are `not_modelled` with `none`.
  `attention_config.indexer_kv_dtype` is `modelled`: the engine reads it and prices the indexer's key cache.
  On DeepSeek V4 and V4.1 `auto` and `fp8` give the 132-byte record, as vLLM resolves them, and `bf16`, `nvfp4`
  and `mxfp4` are refused with vLLM's reason or the kernel's (`mxfp4`'s 68-byte record is not what the graph
  prices); on MiniMax-M3 `auto` and `bf16` size the indexer cache and any other value is refused; a model with
  no indexer cache refuses every value but `auto` (blis-vllm#84). The covered class is `auto` and `fp8`.
- **Value classes** (VLM-3) are in `fields/value_classes.json`, generated from the same
  pinned source by `fields/gen_classes.py` for every `modelled` and `proven_inert` field: for a
  boolean or enumerated field the values (a value some comparison singles out is a class, the
  rest are one class); for a numeric field the domain (the field's own `ge`/`gt`/`le`/`lt`) cut at
  every number the source compares it with. Each class cites the file and line of the comparisons
  that form it and lists the traces whose scenario states a value in it (`covered_by`; a key a scenario leaves out runs the runner's default and is not counted). The
  generator finds a comparison where the field, or a name the same function assigned from it, meets a
  literal; it does not follow a value passed to another function, which `fields/classes_extra.json`
  covers one site at a time with a reason, and the file's `limits` says so. `fields/test_classes.py`
  fails when a modelled field has no classes, a class is backed by no comparison, or a cited line no
  longer holds it; `TestEveryModelledFieldRecordsItsValueClasses` fails the same way without the
  source. `fields.json` carries the classes themselves (the substrate's loader refuses a modelled
  or proven-inert field without them): `fields/gen_fields.py --value-classes fields/value_classes.json`
  writes them there. `fields/value_classes.json` stays as the generated basis (the comparisons and traces behind each class), which the tests read, and
  `TestEveryModelledFieldRecordsItsValueClasses` fails if the two disagree.
- **Value classes under-approximate.** A class edge is a comparison of the field with a literal that the
  generator finds in the pinned source. It does not find a field read by name through a helper that takes
  the name as a string (`_scheduler_config_int(vllm_config, "max_num_batched_tokens", 0)`,
  `vllm/models/deepseek_v4/common/ops/cache_utils.py:812-814@5df63a72`), a comparison with
  another field (listed under `relations`, 25 of them for `max_num_batched_tokens`), or a threshold made by
  `min`, `range` or arithmetic (`max_cudagraph_capture_size` also depends on `min(max_num_tokens, ...)`,
  `vllm/config/vllm.py:2446@5df63a72`). So a field with one class, `max_num_seqs` for one, is a field no
  scanned comparison singles out, not a field the engine treats alike at every value, and a trace that covers
  a class shows the port matches vLLM at the values that trace states, not at every value of the class. The
  `limits` text of `fields/value_classes.json` says the same.
- **Inertness proofs** (design §7.4, tier T1) are `oracle/inertness/<field>.json`, written by
  `oracle/inertness.py` and named by the field's `proof`. The harness varies the field over one value
  per value class and runs vLLM's own code: the real Scheduler on oracle scenarios (the scenario's
  `scheduler_config` states the probe), `EngineArgs.create_engine_config()` (every field of the resolved
  scheduler, cache, parallel and compilation config is compared across the probes except the field
  itself), and for a front-end field the real `OutputProcessor` (the requests it hands back to the engine
  core to abort are compared). A field is inert only if every run's decisions to the engine are identical
  across the probes and every Python read of the field (found by `fields/gen_classes.py --reads`, plus
  reviewed reads the scan cannot follow) has a disposition that holds: a line executed in the named run
  (coverage is kept per run, and matches a file by its path inside vLLM so it counts the installed wheel CI
  runs as well as a checkout), or a keyword passed on to a callee that has its own. The other cases §7.4
  admits (a surface the engine refuses by name, a platform check false on CUDA) are not accepted by the
  harness yet; a proof naming one fails as an unknown disposition. A front-end effect is recorded
  (`frontend_effect`), not hidden. The read scan is of Python. A proof may list `named_in`, every file of
  `vllm/` and `rust/` that names the field, and the check fails when the tree names it in a file the proof
  does not list; that is how the Rust front end's reads of `stream_interval` (`rust/src/cmd/src/cli.rs:277`,
  `rust/src/llm/src/lib.rs:97-102`, `rust/src/engine-core-client/src/client/output_channel.rs`) are listed
  and reviewed without being run, and how the proof shows no file of the engine names it.
  A variable no file of the serving process names has a proof of kind `unread`. That is a reviewed
  static fact (`"argument": "reviewed_static_fact"`), not a run: the scan shows no file names the
  variable, and what the proof says about why it cannot matter is an argument the harness does not check
  beyond what its `refusal_tests` name. The three `MC_` variables belong to the Mooncake transfer-engine
  library, which vLLM loads only through a Mooncake connector; a deployment selects one by connector name
  or through NIXL's `kv_connector_extra_config` backends, and the Go tests the proofs name
  (`TestAMooncakeConnectorNameIsRefused`, `TestTheConnectorBlockCannotStateWhichBackendNixlLoads`,
  `TestNoMooncakeConnectorIsServed`) pin that the engine refuses both.
  `TestEveryProvenInertFieldHasAPassingProof` holds the list to the files, to the value classes (the
  probes reach every class), to the engine's Go code (no non-test file but the listed ones names the
  field) and to the refusal tests (each still exists). `oracle/test_inertness.py` (about 5 minutes
  with the pinned source) shows the harness failing a field that changes the schedule
  (`enable_chunked_prefill` off), a config run and an output-processor run whose engine-facing result
  changes, a read it was not told about, a stale read, a run that did not execute the line it is credited
  with (including when an earlier run did), a file that names the field and is not listed, and a variable the
  source names.
  `stream_interval` is proven inert for the engine only. The simulator's delivery stage does not read it
  yet (inference-sim#2035 is open), so a recorded point that states 20 or 32 is simulated without the
  grouping its client saw, and the field is not G1-eligible until that stage reads it.
- **Narrower issues.** Part of what some `not_modelled` fields need is now its own issue; the fields keep
  the citations they had (the fields.json entry and its status are unchanged here):
  `api_server_count` (blis-vllm#105), the synthetic `rejection_sample_method` (#106),
  `default_chat_template_kwargs` and thinking mode (#107), `enable_eplb` (#108), and decode context
  parallelism and `prefix_match_unit` (#109). `api_server_count` has the estimation rule `none` and is
  not delivered: stating it changes nothing the engine simulates only for the single-client data-parallel
  case, because vLLM scales the in-flight routing score, the optimistic increment and each client's scan
  start by the client count (`vllm/v1/engine/core_client.py:1639-1640,1671,1685@5df63a72`); #105 asks for
  that model.
- **`inert_unproven` is a transition status**, allowed only until M2 (release plan §3.3). The
  perturbation test over every `modelled` field, VLM-3's other check, is not here yet.
- `TestTheFieldListLoadsAndNamesThisRelease` loads the file with the substrate's strict loader,
  checks it names this release, is in canonical form, and that every cited trace exists.

## Licence

Apache-2.0. Files translated from vLLM keep vLLM's licence and name the upstream file and commit
they were derived from; see [NOTICE](NOTICE).
