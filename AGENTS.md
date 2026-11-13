# Agent instructions for inference-sim

The operating context for an agent working in this repository is in `CLAUDE.md`. The block below is
generated; do not edit it by hand.

<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->
## Invariants checked in inference-sim

Generated from the invariant registry, `blis-substrate/contracts/invariants`, at the
substrate version this repository pins (design §13.1). Each line is an invariant this
repository runs a check for: its identifier, the first sentence of its statement, its
status, and each of its checks here. `ci` means the check runs in CI; `new` means it does not
yet, and the target milestone says when it must. A pull request cites the identifiers it
keeps or introduces; CI refuses an identifier that does not exist (X14).

- **X1** (partial; target ci by M4) What a user would hand llm-d, vLLM or SGLang is what they hand BLIS, in the upstream's own field names, types, defaults and errors.
  - `ci`, pull_request: The scenario's router flags, configuration file and manifests reach the router unchanged, and the router parses them with its own code (`cmd: TestScenarioRouterInputs_MapsTheRouterBlock`)
- **X2** (new; target ci by M4) Outside its tests, blis-substrate contains no engine name, no engine version and no engine's policy.
  - `new`, pull_request: PoolPlan, StepShape and Transfer contain no field whose values come from an engine's vocabulary, and a zero value means "absent" or the field's documented neutral value (such as stage 0), never an engine default (a reflection test). Separately, a scenario both engines support is priced from each engine's plan, and every price difference is traced to a named difference between the plans
- **X3** (partial; target ci by M4) Scheduling, KV management, KV connectors and tiering decisions, default resolution, and lowering a model to kernel implementations belong to blis-vllm or blis-sglang.
  - `ci`, pull_request: The simulator reaches engines only through engine.Engine, at one build site (`cmd: TestTheBinaryReachesBlisVLLMOnlyThroughTheBuildSite`)
  - `new`, pull_request: Scheduler and KV types exist only in engine modules
  - `new`, pull_request: The simulator does no resource arithmetic (KV blocks, bytes, tier capacities); it asks the engine or the cost model. The one exception is dividing a link's bandwidth among transfers and the cross-domain traffic of step collectives that overlap in time (SIM-13) (§6.2)
- **X4** (partial; target ci by M3) The fork changes only how the router is hosted: how it reads time, starts work, draws random numbers and makes calls.
  - `new`, pull_request: Under BLIS's simulated runtime, every decision is one the default runtime can make from the same recorded state. Each decision is replayed from the recorded state just before it, not from the simulation's own history, so one different tie-break cannot make every later decision incomparable. Ties and random draws are compared as sets of the choices the upstream could make
  - `ci`, pull_request: BLIS imports only the fork's pkg/embed.New from the router (`sim/llmd: TestRouterImportsOnlyTheEmbedConstructor`)
  - `ci`, pull_request: Only audited plugin types are reachable: every type BLIS supports is hosted by the fork's audited list (RTR-4) (`sim/llmd: TestSupportedTypesAreHostedByTheFork`)
- **X6** (partial; target ci by M3) For fixed scenario bytes, seed, module versions, Go toolchain version and catalog and coefficient registry content hashes, the simulator's standard output is identical byte for byte across runs, thread counts and supported platforms.
  - `ci`, pull_request: Identical output across runs and thread counts; a recorded trace replays identically (`sim/llmd: TestDeterminism_AcrossProcesses`)
  - `new`, pull_request: Identical output on linux/amd64, linux/arm64 and darwin/arm64: nightly, and on every pull request that changes kernel/ or an engine's lowering. Two things differ by platform. First, the Go compiler may fuse x*y+z into one fused multiply-add on some architectures and not others, which changes the last bits; so cost-model and engine code writes float64(x*y) + z where fusion must not happen, or calls math.FMA where it must. Second, some functions in Go's math package, such as Exp and Log, use assembly or CPU-dependent code on some platforms, so their last bits can differ between machines; cost-model and engine code calls the pure-Go copies in contracts/check/fmath instead, which use no fused multiply-add. A lint enforces both rules. stack.json records the toolchain version (SIM-3)
  - `new`, pull_request: No unsorted map iteration on an output path, and no unnamed randomness. In the router fork this lint covers files the fork wrote; upstream code is covered by the differential test in X4
- **X7** (partial; target ci by M4) Contracts live in blis-substrate/contracts.
  - `new`, pull_request: A lower repository refers to accuracy-dashboard only through the harness pin file
  - `new`, pull_request: Every generated file matches its generator's output
- **X8** (new; target ci by M2) Every number records the exact versions and data it came from, and can be regenerated from them.
  - `new`, pull_request: Every Result records the module versions, the source and content hash of the catalog and the registry, any overlay hashes, and whether the run used released data
  - `new`, pull_request: An accuracy change's success criteria are committed before any commit that reads its held-out data
- **X9** (new; target ci by M3) For every pinned upstream release, each feature the upstream offers has exactly one status in a generated list.
  - `new`, pull_request: Coverage of the llm-d stack's published configurations is computed and published per release
- **X11** (partial; target ci by M1) Pins are exact release tags.
  - `new`, pull_request: Selected-version lint (LIF-2), in every repository: no replace outside inference-sim
  - `new`, pull_request: In inference-sim, exactly one, from github.com/llm-d/llm-d-router to github.com/inference-sim/llm-d-router at a -blis.N tag
  - `new`, pull_request: No other module requiring github.com/llm-d/llm-d-router
  - `new`, pull_request: Exclude directives only in inference-sim, each naming a plain tag of an engine module
  - `new`, pull_request: No committed go.work
- **X12** (partial; target ci by M1) Only derived values are committed, never third-party data, with one exception: vendor config.json files whose licence is on the allow-list.
  - `new`, pull_request: Name scan in every repository, which also reads the text of every open issue each night
- **X13** (partial; target ci by M1) Every behaviour change has a test that fails without it.
  - `ci`, pull_request: Byte gate in inference-sim (`cmd: TestParity_RunStdout_DeterministicAndEagerLazyIdentical`)
  - `new`, pull_request: The corpus comparison, run by the scoring harness at the pinned digest from each repository's CI, for pull requests declared as refactors
  - `new`, pull_request: Mutation checks: on a pull request, the functions it changes and the functions covered by any test it changes are mutated, and only the tests that cover them run
  - `new`, nightly: The full sweep runs nightly and on release tags
- **X14** (new; target ci by M1) A pull request's description is accurate: the commit hashes it lists are the branch heads, it merges cleanly, and every invariant it cites exists.
  - `new`, pull_request: The description check, a shared workflow run in every repository
- **X16** (new; target ci by M1) A skipped test in a required or gate job is a failure, not a pass, unless it is on the quarantine list of X25.
  - `new`, pull_request: Each repository's CI configuration
- **X17** (partial; target ci by M1) Every repository's protected branch requires its named checks, including formatting and lint.
  - `ci`, practice: The protected branch requires the repository's named checks, including formatting and lint
- **X19** (new; target ci by M2) Scripts and generators, in every language, never write an engine default.
  - `new`, pull_request: The engine-name lint (X2), run on *.py and *.sh in every repository that has scripts
  - `new`, pull_request: In every scenario generator, a test that a generated scenario contains only stated or recorded values, and workload-dependent values with their citation. The name lint cannot see a bare number under a neutral key, so the second test is the one that catches it
- **X20** (new; target ci by after M4) A commit written with an agent's help says so in one Assisted-by: <tool> line, and carries no Co-authored-by line naming the tool.
  - `new`, pull_request: A commit-policy check, required on every pull request in every repository, over the commits the pull request adds: one Assisted-by line, no Co-authored-by or Signed-off-by naming a tool
  - `new`, practice: The person who takes responsibility for a commit signs it and signs it off, or instructs an agent to push it in that person's name. Encouraged, never checked before the closing step after M4
- **X21** (new; target ci by M1) Everything a build or a release runs is pinned and traceable.
  - `new`, pull_request: A workflow lint that refuses an action not pinned by commit
  - `new`, pull_request: Pip install --require-hashes in every Python CI job
  - `new`, release: The release job, which fails without provenance and a bill of materials
  - `new`, pull_request: A check that LICENSE and any required NOTICE exist
- **X22** (partial; target ci by M4) Every verdict (a CI result or the review agent's review, including its answers to review questions) names the git tree hash of the exact tree it judged.
  - `new`, pull_request: The shared pull-request check in blis-substrate/.github/workflows, called by every repository, which records each review-agent answer with the head tree
  - `new`, pull_request: A workflow lint that refuses ignored exit codes
  - `new`, pull_request: Each job's item count
- **X23** (new; target ci by M1) A version is written in one place.
  - `new`, pull_request: A lint in every repository, over every file type, for version literals of BLIS modules or Go that are not generated
- **X24** (new; target ci by M4) Every review finding ends in one of three outcomes before merge: fixed, filed as a linked issue, or rejected with a reason.
  - `new`, pull_request: A pull-request check that refuses a review thread with no outcome
- **X25** (new; target ci by M1) Tests read only their declared inputs, never sibling checkouts, ignored directories or the user's environment.
  - `new`, pull_request: CI in a clean container with no sibling checkouts
  - `new`, pull_request: The quarantine list, which fails CI when an entry expires
  - `new`, pull_request: A required check that fails while main is failing and the pull request does not name the failing test's issue
- **X26** (new; target ci by M1) In documentation, every source path, symbol, issue or pull-request reference, and every statement that a feature is supported, is either generated (from surface.json, stack.json or the code) or checked against the pinned tree.
  - `new`, pull_request: A documentation reference check in every repository
- **X28** (new; target ci by M1) An issue in a BLIS repository that is cited as the reason for a refusal, for a refused or not-modelled item in a field or surface list, or in a code comment that marks a gap or a divergence, stays open.
  - `new`, nightly: A nightly job in every repository that resolves each cited issue and fails on a closed one
  - `new`, pull_request: A pull-request check that refuses a change closing an issue still cited elsewhere
  - `new`, nightly: The nightly job also reports an issue closed outside a merge
- **SUB-14** (new; target ci by M2) The memory breakdown closes.
  - `new`, pull_request: A start-up refusal in inference-sim when a node's host-memory figures exceed its host memory
- **SUB-17** (new; target ci by M3) A transfer or collective names its endpoints by rank, and the cost model prices it over the path between them through the catalog's links: the NVLink domain, the NIC and each switch tier.
  - `new`, pull_request: The link-capacity test of SIM-13
- **RTR-4** (partial; target ci by M1) A plugin type runs in BLIS only after an audit.
  - `ci`, pull_request: TestSupportedTypesAreHostedByTheFork in inference-sim, against the fork's audited list (`sim/llmd: TestSupportedTypesAreHostedByTheFork`)
- **RTR-5** (new; target ci by M4) The llm-d P/D sidecar is hosted under the same runtime interface, so P/D leg handling is llm-d's code rather than a re-implementation in inference-sim.
  - `new`, pull_request: A test in inference-sim that P/D runs go through the hosted sidecar's code, and that the simulator holds no P/D leg logic of its own
- **SIM-1** (new; target ci by M4) The engine is chosen by the scenario's engine: {kind, version}.
  - `new`, pull_request: Tests that build two engine kinds in one binary, and that refuse any scenario using engine_version with a message naming engine: {kind, version}
- **SIM-2** (ci; target reached) No silent degradation: unsupported or ambiguous input is refused with a message; library code returns errors and never exits the process.
  - `ci`, pull_request: Refusal tests (`cmd: TestRunCmd_KernelBackend_ScenarioRefusals`)
- **SIM-3** (new; target ci by M1) Every release publishes stack.json: the exact tag of every BLIS module and every upstream release it was built and tested with, the Go toolchain version, and the llm-d guides commit.
  - `new`, release: A release check that stack.json matches go list -m all, go version and guides.lock for the tagged build
- **SIM-4** (new; target ci by M1) The byte gate (a fixed scenario set whose output must not change in a refactor) runs inside this repository.
  - `new`, pull_request: The byte gate job in inference-sim's CI, on every pull request
- **SIM-5** (new; target ci by M3) Each release reports how much of the llm-d stack's published configurations it can run, per engine (X9).
  - `new`, release: A release job in inference-sim that computes coverage from two inputs, the llm-d guides commit pinned in guides.lock (which stack.json records) and each engine's surface list, and publishes per-item deployment counts through inference-sim coverage. The dashboard adds its blocked-point counts to them (§7.5)
- **SIM-6** (new; target ci by M1) A run that names no catalog or coefficient registry uses the embedded copy.
  - `new`, pull_request: Tests for each case
  - `new`, pull_request: The result records the data source
- **SIM-7** (new; target ci by M2) Every run makes progress.
  - `new`, pull_request: A progress watchdog that fails a run when no request changes state for a set number of engine steps while work is waiting
  - `new`, pull_request: A seeded property test over random pools and prompts, including the exact-fit case and the one-block-over case. The capacity boundary cases are also oracle scenarios in each engine repository
- **SIM-8** (new; target ci by M2) Requests and sessions follow one written state machine, whose end states are the fates of §8.7 (DEV-9).
  - `new`, pull_request: A transition-table test
  - `new`, pull_request: The arrival-count check of DEV-9
- **SIM-9** (partial; target ci by M2) Each refusal has two tests: the smallest configuration that triggers it, and the closest one that does not.
  - `ci`, pull_request: The refusal test pairs in inference-sim, each engine repository and the router fork (`sim/llmd: TestStartupRefusalMessages`)
- **SIM-10** (new; target ci by M1) Any option that changes a byte of the Result is recorded in the Result's inputs.
  - `new`, pull_request: A test per command-line option that toggles it and compares the recorded inputs
- **SIM-11** (new; target ci by M1) Every field the simulator itself reads (scenario keys outside the engine and router blocks, trace headers, replay descriptors) is consumed, refused or inert, and says which.
  - `new`, pull_request: A reflection test over every input schema that toggles each consumed field
  - `new`, pull_request: A parity test that the same invalid value is refused through every input
- **SIM-12** (new; target ci by M2) Each request's tokens are counted once, in a token record the engine emits: prompt tokens, tokens served from the local and external caches, tokens computed, and tokens emitted.
  - `new`, pull_request: A test that every consumer reads the token record
  - `new`, pull_request: Known-answer tests of every capacity function at tensor-parallel and data-parallel widths above one
- **SIM-13** (partial; target ci by M3) The traffic that overlapping transfers and step collectives place on a link never exceeds its capacity.
  - `ci`, pull_request: A property test over random placements and transfers (`sim/cluster: TestPDContention_*`)
  - `new`, pull_request: The same property test with step collectives included
- **SIM-14** (new; target ci by M2) Every cluster control loop that runs beside the router, such as an autoscaler or the placement of LoRA adapters, is a component (§8.1): it acts only on published values (DEV-1, DEV-4).
  - `new`, pull_request: A reflection test that every configurable control policy has an entry in the list of control loops
  - `new`, pull_request: A Result test that each active loop is named with its validation status
- **FID-9** (new; target ci by M2) The engine is tested as the simulator drives it, not only on its own.
  - `new`, pull_request: The coupled replay test in inference-sim's CI, over every engine repository's recorded traces
- **DEV-1** (new; target ci by M2) Components interact only through timestamped messages.
  - `new`, pull_request: In the router fork and inference-sim, unexported constructors for the router's view types, so only the message-delivery code can build them
  - `new`, pull_request: An import-graph test that finds no import of engine-internal packages from router-side packages
  - `new`, pull_request: The coupled replay test (FID-9)
- **DEV-2** (new; target ci by M2) Each instant is processed in the three phases of §8.2, and messages are ordered by (time, class, sender, sender's send count).
  - `new`, pull_request: In inference-sim, a test that permutes the scheduling order of same-instant events from different components and compares the Result
  - `new`, pull_request: An assertion that no two queued messages share a key
  - `new`, pull_request: A run of the corpus under a second priority-queue implementation
- **DEV-3** (new; target ci by M2) Each engine declares, for each kind of external input, the step boundary at which it takes effect, whether it arrives during a step or at the instant one ends, and whether the upstream's order there is fixed or a race between threads.
  - `new`, pull_request: A property test in inference-sim that, under the cost model's timing, every external input takes effect at its declared boundary (FID-9)
- **DEV-4** (new; target ci by M2) No component observes a message before its timestamp.
  - `new`, pull_request: In inference-sim, a trace check that every observation is the latest value published at or before its time minus the declared delay
  - `new`, pull_request: A lint on cache keys of published values
- **DEV-5** (new; target ci by M2) Only finitely many events happen in any finite stretch of simulated time.
  - `new`, pull_request: In inference-sim, a watchdog on events per instant across the whole model
  - `new`, nightly: A nightly scaling run at 104, 105 and 106 requests that fails on growth in events per request or wall time per event
- **DEV-6** (new; target ci by M1) Every random draw comes from a substream named by component, purpose and entity, seeded from the SHA-256 of the scenario seed and the name, by a generator named in the frame.
  - `new`, pull_request: In inference-sim, a test that changes one server setting under a closed-loop session workload and compares every request's prompt and output lengths byte for byte
  - `new`, pull_request: The lint against unnamed randomness (X6)
- **DEV-7** (new; target ci by M1) Every run records its experimental frame (§8.6).
  - `new`, pull_request: In inference-sim, a round-trip property test over every workload feature (closed loop, sessions, accumulated context, P/D, deadlines)
  - `new`, pull_request: Two-sample tests for each converter against its source
- **DEV-8** (new; target ci by M2) Every reported metric names its definition and scope (§8.7), and a comparison is refused unless both sides have the same definition, scope and aggregate.
  - `new`, pull_request: A test that they compare only figures with matching definitions, scopes and aggregates
- **DEV-9** (partial; target ci by M1) Each request's fate is written once, at one site, and every latency sample is derived from it.
  - `new`, pull_request: In inference-sim, the fate type, which fails on a second write
  - `ci`, pull_request: The arrival-count check at the end of every byte-gate scenario (`sim: TestINV1Accounted`)
  - `new`, pull_request: A known-bad scenario for each fate (X18)
- **DEV-10** (new; target ci by M3) A steady-state run is classified as settled, overloaded or incomplete by one detector, and only a settled run gives a latency figure for steady state.
  - `new`, pull_request: The detector in inference-sim, reached by the applications through inference-sim metrics, and its reference tests against single-server queues whose behaviour is known exactly: open M/M/1 (exponential service times) and M/G/1 (general service times) at loads from 0.5 to 1.5 of capacity, including 0.9 to 0.99 where settling is slow, closed queues for fixed-concurrency benchmarks, and heavy-tailed service, which must be classed incomplete at practical window lengths
- **DEV-12** (new; target ci by M1) Each change is a verification change or a validation change by what it does (§8.9), and is judged as such.
  - `new`, pull_request: In every repository, the change class in the pull-request template, and a check that the outputs that moved match the declared class
- **DEV-13** (new; target ci by M2) A request's sampled output length is written by the workload generator and read only by the trace writer, by replay, and by the engine's stop check, where it acts as an end-of-sequence token at that position.
  - `new`, pull_request: A field-access lint in inference-sim and each engine repository that refuses any read of the sampled length outside these declared sites and tests, with a known-bad input (X18)
- **COV-1** (partial; target ci by M1) One scorer produces every number, and it lives only in accuracy-dashboard: no other repository compares a simulation with a measured end-to-end benchmark.
  - `new`, pull_request: A lint that refuses code outside accuracy-dashboard that reads end-to-end benchmark units to compute an error
- **COV-12** (partial; target ci by M3) The scored simulation receives the requests the measured run sent: the same requests, in the same order, with the same prompt lengths, output lengths and shared prefixes.
  - `ci`, pull_request: The single-turn recomputation of the client's seed-0 lengths and the sent-length check, in inference-sim's tools/harness, until they move into the dashboard's harness (`tools/harness: TestEveryMeasuredOrResolvedPointSentItsTarget`)
- **LIF-1** (new; target ci by M1) A tag names one commit forever.
  - `new`, practice: Protected tags
  - `new`, nightly: A nightly check that every tag resolves to its recorded commit
- **LIF-2** (partial; target ci by M1) In every build, each BLIS module resolves to the exact version the release records in stack.json, and the router fork resolves to a -blis.N tag through inference-sim's one replace directive (X11).
  - `new`, pull_request: Selected-version lint (go list -m all), which for the router reads the replace target that go list -m -json reports, not the placeholder version on the require line
  - `new`, pull_request: A check that every module in go list -m -json all declares a GoVersion no higher than contracts/toolchain
  - `new`, pull_request: A CI step that fails if go version differs from it
  - `new`, release: The pre-release test of every engine and router package against the simulator's selected versions (§3.2)
- **LIF-3** (new; target ci by M1) Only the release bot cuts releases, and only from a commit where every ci invariant passed.
  - `new`, practice: Repository settings that let only the release bot and people with write access create tags
  - `new`, nightly: A nightly check that every tag made by a person is annotated and listed in its release notes
- **LIF-6** (new; target ci by M1) Each inference-sim release runs exactly one upstream release of each engine and of llm-d, and lists them in stack.json.
  - `new`, pull_request: A refusal test per engine kind
  - `new`, pull_request: A check that the message's release names match stack.json history
- **LIF-7** (new; target ci by M1) A deprecated feature is removed by the first release after its deprecation that may break compatibility: a MAJOR release, or a MINOR release while the repository is at v0.
  - `new`, pull_request: A whole-program dead-code check (Go's deadcode), run in stack CI's workspace from the main packages of every binary BLIS ships, with test files not counted as reaching code. Upstream files in the router fork, and the interfaces in contracts/, are exempt

### Maintainer practices

No program checks these. A maintainer with write access keeps each one for this
repository as a whole and confirms it when the repository's settings change (design §4.2).

- [ ] **X17** The protected branch requires the repository's named checks, including formatting and lint
- [ ] **X20** The person who takes responsibility for a commit signs it and signs it off, or instructs an agent to push it in that person's name. Encouraged, never checked before the closing step after M4
- [ ] **LIF-1** Protected tags
- [ ] **LIF-3** Repository settings that let only the release bot and people with write access create tags

### Agent guidance

Follow these on every change you write or review. They are not checked by a program
(design §4.2).

- **X24** Answer each review thread with its commit or its issue, and resolve the thread once the reviewer is satisfied.

### Review questions

CI lists each question below in the description of a pull request that changes one of
its paths. It is guidance for the review agent (design §13.2): the agent answers it yes or
no in a review comment, with the files and lines that support the answer. No check waits
on a question.

- **X3** Is engine behaviour, such as scheduling, admission or KV accounting, being added to the simulator instead of an engine repository? Expertise: inference-sim expertise. Applies when: a change to the simulator's request-lifecycle or capacity code. Paths: `sim/engine_driver*.go`, `sim/instance_state.go`, `sim/request.go`, `sim/event_queue.go`, `sim/engine_records.go`, `sim/dp_modes.go`, `sim/cluster/**`, except `**/*_test.go`.
- **X6** Could this floating-point reduction give a different result on another platform? Expertise: expertise in the changed repository. Applies when: a parallel or floating-point reduction is added, or the cross-platform check fails. Added by a reviewer with the label `review:X6`.
- **X7** Does this change give a second home to a fact another repository owns? Expertise: expertise in the changed repository. Applies when: a pull request adds a literal list or table, or a constant that cites another repository. Added by a reviewer with the label `review:X7`.
- **DEV-12** Does this change alter what is modelled? Expertise: expertise in the changed directory. Applies when: a pull request is declared a verification change but moves a Result outside the byte gate. Added by a reviewer with the label `review:DEV-12`.
<!-- END GENERATED: invariants -->
