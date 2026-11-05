# Agent instructions for blis-vllm

These rules apply to every AI-assisted change to this repository. Human contributors follow the
same rules; they are written here so an agent reads them first.

## Cross-repository rules (all BLIS repositories)

blis-vllm sits between blis-substrate and inference-sim. A change that touches more than one
repository follows
[Changes across the repositories](https://github.com/inference-sim/blis-substrate/blob/main/docs/contracts/contributing/cross-repo-changes.md).
In short:

1. **Public artifacts.** Never write internal project, customer or colleague names in a commit,
   code, comment, document, issue or pull request. Refer to private measurements by what they
   are ("internal deployment reports"). Run the name scan before every push.
2. **Data.** Never commit third-party data. Commit derived values only, each with its source URL
   plus commit or sha256.
3. **Versions.** Pin dependencies by release tag, or by the exact pushed head marked `TEMPORARY`
   with its pull request. Never commit a `replace` in `go.mod`.
4. **Tests.** Every behaviour change has a test that fails without it. Mutation-check it:
   revert the fix, see the test fail, restore, and say so in the commit message.

## This repository

- **One release.** A build models one vLLM release (`Upstream` in `provenance.go`). Every
  statement about vLLM cites `vllm/<path>:<lines>@<commit8>` at that release.
- **Ported files.** A Go file translated from vLLM keeps vLLM's licence and records its source:
  ```go
  // SPDX-License-Identifier: Apache-2.0
  // SPDX-FileCopyrightText: Copyright contributors to the vLLM project
  // Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
  ```
  `TestEveryFileStatesItsProvenance` (`architecture_test.go`) enforces it.
- **No clock, no goroutines.** Nothing reads `time.Now`, sleeps or starts a goroutine;
  `TestNoClockNoGoroutines` enforces it.
- **The oracle decides.** Behaviour is checked against vLLM's own classes in CI (`oracle/`).
  The oracle may patch only the GPU, the clock, transports and external services, never a
  decision; every patch is listed in `oracle/mocks.json`, and CI fails on a patch of decision
  code. Never overwrite a golden trace without human review (`docs/testing.md`).
- Tests: `go test ./...`; lint: `golangci-lint run ./...` (v2.14.0); the oracle tiers: the
  `oracle-tier2` CI job, step by step, from a Python with `oracle/requirements.lock` installed
  (`pip install --require-hashes --no-deps -r oracle/requirements-hashed.txt`, Linux x86_64;
  `oracle/README.md` gives each runner's command and the environment variables the Go fuzz
  tests read).
- **Process checks.** `.github/workflows/blis.yml` calls blis-substrate's shared checks by
  commit; `.github/workflows/private-names.yml` is the name scan. The section below and
  `.github/pull_request_template.md` are generated from the invariant registry
  (`go run github.com/inference-sim/blis-substrate/tools/invariants agents -repo blis-vllm`,
  and `template`); CI checks them.

<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->
## Invariants checked in blis-vllm

Generated from the invariant registry, `blis-substrate/contracts/invariants`, at the
substrate version this repository pins (design §13.1). Each line is an invariant this
repository runs a check for: its identifier, the first sentence of its statement, its
status, and each of its checks here. `ci` means the check runs in CI; `new` means it does not
yet, and the target milestone says when it must. A pull request cites the identifiers it
keeps or introduces; CI refuses an identifier that does not exist (X14).

- **X1** (partial; target ci by M4) What a user would hand llm-d, vLLM or SGLang is what they hand BLIS, in the upstream's own field names, types, defaults and errors.
  - `new`, pull_request: Every configuration field of the pinned release is classified as modelled, proven inert, or not modelled (VLM-3, SGL-3)
  - `ci`, pull_request: An unknown field, or a not-modelled field set away from its default, is refused with its upstream name (`rules/engineconfig: TestDecodeRefusesAnUnknownKey`)
  - `ci`, pull_request: Defaults match: the upstream's own configuration code resolves each oracle scenario, and the port's values are compared field by field (`ci.yml job packs-against-vllm: rules/v0_32 -run MatchSource`)
  - `new`, pull_request: Every configuration the upstream rejects is rejected. This is tested on a generated set: each field's boundary values, taken from the upstream's own argument and validation metadata, and one case for each branch of the upstream's validation code, with branch coverage required
- **X3** (partial; target ci by M4) Scheduling, KV management, KV connectors and tiering decisions, default resolution, and lowering a model to kernel implementations belong to blis-vllm or blis-sglang.
  - `new`, pull_request: No engine module imports another
  - `new`, pull_request: Every ported file names the release that the repository's Upstream record names
- **X5** (partial; target ci by M3) Each engine repository runs the upstream's own Python code on the same inputs and compares its decisions with the Go port's, step by step.
  - `ci`, pull_request: Every replacement the test harness makes is listed in the manifest; a changed manifest carries a new version (`oracle/check_mocks.py oracle/mocks.json`)
  - `ci`, pull_request: Decisions match on every recorded scenario (`core/sched: TestDifferentialAgainstVLLM`)
  - `ci`, pull_request: A recorded trace's decisions change only in a commit marked as reviewed (`oracle/check_goldens.py`)
  - `new`, pull_request: Every ported function is exercised by some recorded scenario, and deliberately broken copies of it are caught
- **X6** (partial; target ci by M3) For fixed scenario bytes, seed, module versions, Go toolchain version and catalog and registry content hashes, the simulator's standard output is identical byte for byte across runs, thread counts and supported platforms.
  - `ci`, pull_request: No clock reads and no goroutines in engine or cost-model code (`TestNoClockNoGoroutines`)
  - `new`, pull_request: No unsorted map iteration on an output path, and no unnamed randomness. In the router fork this lint covers files the fork wrote; upstream code is covered by the differential test in X4
- **X7** (partial; target ci by M4) Contracts live in blis-substrate/contracts.
  - `new`, pull_request: A lower repository refers to accuracy-dashboard only through the harness pin file
  - `new`, pull_request: The engine repositories and the router fork import only substrate packages under contracts/
  - `ci`, pull_request: Every generated file matches its generator's output (`go run ./cmd/metricsmanifest -check`)
- **X8** (new; target ci by M1) Every number records the exact versions and data it came from, and can be regenerated from them.
  - `new`, pull_request: An accuracy change's success criteria are committed before any commit that reads its held-out data
- **X9** (new; target ci by M3) For every pinned upstream release, each feature the upstream offers has exactly one status in a generated list.
  - `new`, pull_request: The surface list is regenerated from the pinned upstream source and matches the committed list
  - `new`, pull_request: Every item has a status, and every modelled item names its evidence
  - `new`, pull_request: Every out-of-scope item names one of the three allowed reasons and cites upstream. An engine release does not lower the weighted share of modelled items, and does not move an item from modelled to refused without a fidelity issue that explains why
- **X11** (partial; target ci by M1) Pins are exact release tags.
  - `new`, pull_request: Selected-version lint (LIF-2), in every repository: no replace outside inference-sim
  - `new`, pull_request: No other module requiring github.com/llm-d/llm-d-router
  - `new`, pull_request: No committed go.work
- **X12** (partial; target ci by M1) Only derived values are committed, never third-party data, with one exception: vendor config.json files whose licence is on the allow-list.
  - `new`, pull_request: Name scan in every repository, which also reads the text of every open issue each night
- **X13** (partial; target ci by M1) Every behaviour change has a test that fails without it.
  - `new`, pull_request: The corpus comparison, run by the scoring harness at the pinned digest from each repository's CI, for pull requests declared as refactors
  - `new`, pull_request: Mutation checks: on a pull request, the functions it changes and the functions covered by any test it changes are mutated, and only the tests that cover them run
  - `new`, nightly: The full sweep runs nightly and on release tags
- **X14** (new; target ci by M1) A pull request's description is accurate: the commit hashes it lists are the branch heads, it merges cleanly, and every invariant it cites exists.
  - `new`, pull_request: The description check, a shared workflow run in every repository
- **X16** (new; target ci by M1) A skipped test in a required or gate job is a failure, not a pass, unless it is on the quarantine list of X25.
  - `new`, pull_request: Each repository's CI configuration
- **X17** (partial; target ci by M1) Every repository's protected branch requires its named checks, including formatting and lint.
  - `ci`, settings: The protected branch requires the repository's named checks, including formatting and lint (`branch protection on main`)
- **X19** (new; target ci by M2) Scripts and generators, in every language, never write an engine default.
  - `new`, pull_request: The engine-name lint (X2), run on *.py and *.sh in every repository that has scripts
- **X20** (partial; target ci by M1) Every commit written for BLIS is signed.
  - `new`, pull_request: A commit-policy check, required on every pull request in every repository, over the commits the pull request adds
- **X21** (new; target ci by M1) Everything a build or a release runs is pinned and traceable.
  - `new`, pull_request: A workflow lint that refuses an action not pinned by commit
  - `new`, pull_request: Pip install --require-hashes in every Python CI job
  - `new`, pull_request: A check that LICENSE and any required NOTICE exist
  - `new`, pull_request: The provenance test checks each derived file's licence header
- **X22** (partial; target ci by M1) Every verdict (a CI result, an agent's review, or a person's answer to a review question) names the git tree hash of the exact tree it judged.
  - `new`, pull_request: The shared merge check in blis-substrate/.github/workflows, called by every repository, which compares verdict tree hashes
  - `new`, pull_request: A workflow lint that refuses ignored exit codes
  - `new`, pull_request: Each job's item count
- **X23** (new; target ci by M1) A version is written in one place.
  - `new`, pull_request: A lint in every repository, over every file type, for version literals of BLIS modules or Go that are not generated
- **X24** (new; target ci by M1) Every review finding ends in one of three outcomes before merge: fixed, filed as a linked issue, or rejected with a reason.
  - `new`, pull_request: The shared merge check, which refuses a merge while a review thread has no outcome
- **X25** (new; target ci by M1) Tests read only their declared inputs, never sibling checkouts, ignored directories or the user's environment.
  - `new`, pull_request: CI in a clean container with no sibling checkouts
  - `new`, pull_request: The quarantine list, which fails CI when an entry expires
  - `new`, pull_request: A required check that fails while main is failing and the pull request does not name the failing test's issue
- **X26** (new; target ci by M1) In documentation, every source path, symbol, issue or pull-request reference, and every statement that a feature is supported, is either generated (from surface.json, stack.json or the code) or checked against the pinned tree.
  - `new`, pull_request: A documentation reference check in every repository
- **X28** (new; target ci by M1) An issue in a BLIS repository that is cited as the reason for a refusal, for a refused or not-modelled item in a field or surface list, or in a code comment that marks a gap or a divergence, stays open.
  - `new`, nightly: A nightly job in every repository that resolves each cited issue and fails on a closed one
  - `new`, pull_request: A merge check that refuses a change closing an issue still cited elsewhere
  - `new`, nightly: The nightly job also reports an issue closed outside a merge
- **SUB-3** (partial; target ci by M4) The catalog holds declared facts only: figures read from a published source, each with a citation a reader can follow.
  - `new`, pull_request: A lint for model-name literals in kernel/ and in engine lowering code. Review: that each citation supports its figure
- **SUB-10** (new; target ci by M2) contracts/conformance includes a set of unusual model shapes: every layer windowed, every layer recurrent, MLA only, a draft layer declared last, a single layer, and no attention at all.
  - `new`, pull_request: The conformance suite, run in blis-substrate, blis-vllm and blis-sglang
- **SUB-14** (new; target ci by M2) The memory breakdown closes.
  - `new`, pull_request: In each engine repository, oracle tier T3 against the upstream's own buffer-sizing functions
- **SUB-20** (new; target ci by M2) No price, size or engine decision depends on a free-text label.
  - `new`, pull_request: A syntax-tree lint in blis-substrate and each engine repository that refuses a comparison on a documentation-only string field outside tests
- **VLM-1** (ci; target reached) Decisions match vLLM at the pinned commit, step by step, on every oracle scenario.
  - `ci`, pull_request: Oracle tiers T0–T6 (§7.4) (`ci.yml jobs packs-against-vllm, oracle-tier2, metrics-manifest`)
- **VLM-2** (partial; target ci by M4) Every Go file states where it came from: derived from a vLLM path at a release and commit, moved from another repository, or original.
  - `ci`, pull_request: Provenance test: every file states its provenance (`TestEveryFileStatesItsProvenance`)
  - `new`, pull_request: Provenance test: no file names a vLLM release other than the one the Upstream record names
- **VLM-3** (new; target ci by M2) fields.json classifies every configuration field and every environment variable the release reads, including those read by the communication libraries it uses (NCCL_*, NVSHMEM_*, UCX_*), as modelled, proven inert (with a passing proof), or not modelled (with an issue).
  - `new`, pull_request: Completeness check
  - `new`, pull_request: Inertness proofs (§7.4)
  - `new`, pull_request: A perturbation test over every field marked modelled
  - `new`, pull_request: A check that every field has an estimation rule or says it has none. A second check fails on any field whose type can hold a class or module path and is not marked as able to name code
- **VLM-4** (new; target ci by M2) surface.json lists every connector, offload tier, KV-cache spec type and its manager, speculative method, scheduler policy, metric family and KV-event type the pinned release registers, each with a status (X9).
  - `new`, pull_request: Generator against vLLM's registries (§7.5)
- **VLM-5** (ci; target reached) No clock reads and no goroutines.
  - `ci`, pull_request: Syntax-tree test (`TestNoClockNoGoroutines`)
- **VLM-6** (new; target ci by M2) A value of an enumerated setting, such as a connector, backend or policy name, runs only through the surface item that models it.
  - `new`, pull_request: Dispatch generated from surface.json
  - `new`, pull_request: A perturbation test per enumerated value
- **VLM-7** (new; target ci by M2) Every KV-cache spec type and group role the pinned release registers in KVCacheSpecRegistry has a status in surface.json that X9 allows.
  - `new`, pull_request: Generator against KVCacheSpecRegistry (§7.5)
  - `new`, pull_request: A T2 scenario per ported manager
  - `new`, pull_request: A start-up refusal test with a model that needs each refused spec type
- **SIM-8** (new; target ci by M1) Requests and sessions follow one written state machine, whose end states are the fates of §8.7 (DEV-9).
  - `new`, pull_request: An oracle scenario for identifier reuse in each engine repository
- **SIM-9** (partial; target ci by M2) Each refusal has two tests: the smallest configuration that triggers it, and the closest one that does not.
  - `new`, pull_request: The refusal test pairs in inference-sim, each engine repository and the router fork
- **FID-1** (partial; target ci by M2) The manifest lists every replacement.
  - `ci`, pull_request: Oracle/check_mocks.py, from contracts/harness, in each engine repository (`oracle/check_mocks.py`)
- **FID-2** (partial; target ci by M3) The oracle runs in a locked Python environment at the pinned upstream commit, and every recorded trace records that commit, the lock file's hash, the scenario's hash, the seed and the manifest version.
  - `ci`, pull_request: Oracle/check_lock.py and the trace provenance check (`oracle/check_lock.py`)
- **FID-3** (partial; target ci by M3) A recorded trace's decisions change only in a commit carrying a golden-reviewed: line that names the trace and gives the reason.
  - `ci`, pull_request: Oracle/check_goldens.py (`oracle/check_goldens.py`)
  - `new`, pull_request: The oracle runners refuse to write traces without the update flag
- **FID-4** (new; target ci by M2) Every ported function is exercised by at least one recorded scenario.
  - `new`, pull_request: Coverage gate
- **FID-5** (new; target ci by M3) A framework (connector base, offload tiering, HiCache) is marked modelled in the surface list only when its contract has traces of its own, independent of any single connector built on it.
  - `new`, pull_request: A surface-list check that a framework marked modelled names traces of its own
- **FID-6** (new; target ci by M2) A property asserted about an engine port, such as a monotonicity law, is first run against the upstream oracle.
  - `new`, pull_request: The property tests run in the oracle tier before they run against the port
- **FID-7** (new; target ci by M3) For every catalog model and each supported layout, the resident weights an engine reports in its PoolPlan (per-rank shape, stored data type and sharding, after the engine's weight post-processing) equal what the pinned upstream allocates.
  - `new`, pull_request: Oracle tier T0 in each engine repository, run whenever a substrate release changes the catalog or the engine moves to a new upstream release
- **FID-8** (new; target ci by M2) The harness may not contain a copy of an upstream decision function.
  - `new`, pull_request: Oracle/check_mocks.py
- **FID-10** (new; target ci by M2) Every compared field and metric series takes its neutral value and at least one other value somewhere in the scenario set, so a comparison of it can fail.
  - `new`, pull_request: A value-coverage report in each engine repository's CI, which fails on any field or series seen only at one value
- **FID-11** (new; target ci by M3) The work a step shape asks the cost model to price equals what the pinned upstream launches for the same step, for each layer: the rows of every matrix multiply, each operation's FLOPs and bytes, the token count of every draft pass, and the context entries each context-dependent operation reads, bounded by its window where it has one.
  - `new`, pull_request: An oracle comparison, run with T0 and on engine pull requests that change how step shapes are emitted. It executes the upstream forward pass on meta tensors, at the padded batch the upstream launches, under a PyTorch dispatch mode that counts each operation's rows, FLOPs and bytes, with the ranks under a fake process group. Each opaque custom operation, such as attention or a fused mixture of experts, is counted by a rule the manifest lists. Routed-expert rows are compared in total (tokens times experts per token), since routing depends on data. Context reads are counted from the upstream's own attention metadata: sequence lengths, block tables and windows. Known-bad inputs: a matrix multiply priced at fewer rows per token than the upstream runs, and a windowed gather priced over the whole context
- **DEV-3** (new; target ci by M2) Each engine declares, for each kind of external input, the step boundary at which it takes effect, whether it arrives during a step or at the instant one ends, and whether the upstream's order there is fixed or a race between threads.
  - `new`, pull_request: The oracle, which records where the upstream's loop reads its input queue (each read site is listed in the manifest; where that loop is transcribed, as vLLM's data-parallel busy loop is, the check runs against the transcription, FID-8) and checks that reads happen only at the declared boundaries
- **DEV-12** (new; target ci by M1) Each change is a verification change or a validation change by what it does (§8.9), and is judged as such.
  - `new`, pull_request: In every repository, the change class in the pull-request template, and a check that the outputs that moved match the declared class
- **DEV-13** (new; target ci by M2) A request's sampled output length is written by the workload generator and read only by the trace writer, by replay, and by the engine's stop check, where it acts as an end-of-sequence token at that position.
  - `new`, pull_request: A field-access lint in inference-sim and each engine repository that refuses any read of the sampled length outside these declared sites and tests, with a known-bad input (X18)
- **COV-1** (partial; target ci by M1) One scorer produces every number, and it lives only in accuracy-dashboard: no other repository compares a simulation with a measured end-to-end benchmark.
  - `new`, pull_request: A lint that refuses code outside accuracy-dashboard that reads end-to-end benchmark units to compute an error
- **LIF-1** (new; target ci by M1) A tag names one commit forever.
  - `new`, settings: Protected tags
  - `new`, nightly: A nightly check that every tag resolves to its recorded commit
- **LIF-2** (new; target ci by M1) In every build, each BLIS module resolves to the exact version the release records in stack.json, and the router fork resolves to a -blis.N tag through inference-sim's one replace directive (X11).
  - `new`, pull_request: Selected-version lint (go list -m all), which for the router reads the replace target that go list -m -json reports, not the placeholder version on the require line
  - `new`, pull_request: A check that every module in go list -m -json all declares a GoVersion no higher than contracts/toolchain
  - `new`, pull_request: A CI step that fails if go version differs from it
- **LIF-3** (new; target ci by M1) Only the release bot cuts releases, and only from a commit where every ci invariant passed.
  - `new`, settings: Repository settings that let only the release bot and release owners create tags
  - `new`, nightly: A nightly check that every tag made by a person is signed and listed in its release notes
- **LIF-5** (partial; target ci by M4) An engine module, or the router fork, models one upstream release.
  - `new`, pull_request: A test that lists the dependencies of the engine's entry package with go list -deps and fails on any package kept for another release
  - `new`, pull_request: In engine modules, the provenance test, which refuses a ported file that names another release

### Review questions

CI adds each question below to a pull request that changes one of its paths, as a
required checklist item. A code owner answers it yes or no with a one-line reason; an agent
may propose an answer but never gives one (design §13.7).

- **X5** Does this new replacement stand in only for the GPU, clock, transport, environment settings, the hardware platform or an outside service, and not for a decision? If it reproduces a decision, is it declared as a transcription? Reviewer: engine repository owners. When: the manifest changes. Paths: `oracle/mocks.json`.
- **X6** Could this floating-point reduction give a different result on another platform? Reviewer: owners of the changed repository. When: a parallel or floating-point reduction is added, or the cross-platform check fails. Added by a code owner with the label `review:X6`.
- **X7** Does this change give a second home to a fact another repository owns? Reviewer: owners of the changed repository. When: a pull request adds a literal list or table, or a constant that cites another repository. Added by a code owner with the label `review:X7`.
- **X9** Is this "out of scope" reason one of the three allowed, with a citation that supports it, and does the llm-d stack never use the feature on the hardware BLIS targets? Reviewer: engine repository owners. When: an item is marked out of scope. Paths: `surface.json`.
- **DEV-12** Does this change alter what is modelled? Reviewer: owners of the changed directory. When: a pull request is declared a verification change but moves a Result outside the byte gate. Added by a code owner with the label `review:DEV-12`.
<!-- END GENERATED: invariants -->
