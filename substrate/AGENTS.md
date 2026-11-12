# Agent instructions for blis-substrate

These rules apply to every change to this repository, by a person or an agent. They are written
here so that an agent reads them first. The rules come from the BLIS target design; each names
its invariant identifier.

## What belongs here

blis-substrate holds what is true whatever engine runs (design §2.2). Before adding a fact, ask
whether a second engine would need the same edit. If it would not, the change belongs in an
engine repository.

- Outside tests, no engine name, engine version or engine policy (X2).
- The cost model resolves nothing: every default, backend choice, padding and capture decision
  arrives in the plan or the step shape (SUB-6).
- The catalog holds declared facts, each with a citation. A fitted or assumed value is a
  coefficient, and belongs in the coefficient registry (SUB-3, SUB-18).
- Every coefficient is written by its family's writer script, never by hand (SUB-4).
- Packages under `contracts/` import only the Go standard library, the YAML parser and each
  other, and change only compatibly (SUB-1).
- One Go module: no `go.mod` below the root (SUB-13).

## Before a pull request

Run the checks for every directory the change touches:

```sh
gofmt -l . && go vet ./... && go test ./...
go run ./validate/cmd/validate-catalog catalog
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check && python3 -m pytest tests -q)
(cd registry && python3 -m pytest validator -q)
DOCS_RELEASE=main mkdocs build --strict
go run ./tools/invariants render -check
go run ./tools/prcheck workflows
go run ./tools/prcheck versions
go run ./tools/prcheck docrefs
```

- Every behaviour change has a test that fails without it (X13).
- A new or changed data field ships with its validator, a failing example for each way it can
  be wrong, and regenerated reference documentation: `go run ./tools/docgen` (SUB-2).
- Never commit third-party data; a derived value cites its source by URL and commit or hash
  (X12). Never write a private name in a file, commit or pull request (X12).

## A change here must also update

- The documentation that describes what the change removes, renames or alters. A path, symbol,
  issue reference or supported-feature statement in a page is generated or checked (X26).
- The generated copies, with `go run ./tools/invariants render`, when an invariant registry record
  changes; and the record itself, when a check is added or a status moves (design §13.1).
- The reference pages, with `go run ./tools/docgen`, when a schema or data field changes.
- The issue the change closes, named on a `Closes` line; an issue cited as a gap stays open until
  the change that removes its last citation (X28).

## Agents may not

Edit a generated file by hand; weaken, skip or delete a check; change a success criterion after
reading held-out data, or the ledger's cut-points or partition version; approve a merge; cut a
release; review a change they wrote themselves (design §13.7). An agent may act as the review
agent: it answers each review question below in a review comment, with the files and lines that
support the answer.

## Where to look next

The design is the target this repository implements; each rule above names its invariant
identifier, which the invariant registry (`contracts/invariants`) and the appendix at
`docs/contracts/invariants` define. Contributor guides are under `docs/contributing` and
`docs/contracts/contributing`.

<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->
## Invariants checked in blis-substrate

Generated from the invariant registry, `blis-substrate/contracts/invariants`, at the
substrate version this repository pins (design §13.1). Each line is an invariant this
repository runs a check for: its identifier, the first sentence of its statement, its
status, and each of its checks here. `ci` means the check runs in CI; `new` means it does not
yet, and the target milestone says when it must. A pull request cites the identifiers it
keeps or introduces; CI refuses an identifier that does not exist (X14).

- **X1** (partial; target ci by M4) What a user would hand llm-d, vLLM or SGLang is what they hand BLIS, in the upstream's own field names, types, defaults and errors.
  - `new`, pull_request: Every scenario key outside the engine and router blocks is on a fixed list of environment facts
- **X2** (new; target ci by M4) Outside its tests, blis-substrate contains no engine name, no engine version and no engine's policy.
  - `new`, pull_request: No engine identifier in non-test Go identifiers, string literals or registry keys: engine names, engine version strings, upstream source-path citations, and engine-only setting names. The word list is committed in the substrate (invariants/engine-words.txt). A bot regenerates it from each engine release's field list in a pull request, leaving out words the architecture vocabulary also uses, such as dtype or quantization. The count per directory may only go down, measured against the word list of the same version, until it reaches zero
  - `new`, pull_request: The cost model's public functions accept only plans, step shapes, transfers, and catalog and registry handles
  - `new`, pull_request: Coefficient scope keys come from a fixed set with no engine key
  - `new`, pull_request: Model graphs use only architecture vocabulary
- **X3** (partial; target ci by M4) Scheduling, KV management, KV connectors and tiering decisions, default resolution, and lowering a model to kernel implementations belong to blis-vllm or blis-sglang.
  - `ci`, pull_request: The cost model imports no engine module (`kernel: TestTheLibraryImportsNoEngineModel`)
  - `new`, pull_request: Scheduler and KV types exist only in engine modules
- **X6** (partial; target ci by M3) For fixed scenario bytes, seed, module versions, Go toolchain version and catalog and coefficient registry content hashes, the simulator's standard output is identical byte for byte across runs, thread counts and supported platforms.
  - `new`, pull_request: No clock reads and no goroutines in engine or cost-model code
  - `new`, pull_request: No unsorted map iteration on an output path, and no unnamed randomness. In the router fork this lint covers files the fork wrote; upstream code is covered by the differential test in X4
- **X7** (partial; target ci by M4) Contracts live in blis-substrate/contracts.
  - `new`, pull_request: A lower repository refers to accuracy-dashboard only through the harness pin file
  - `new`, pull_request: The module graph across all repositories follows Figure 1
  - `ci`, pull_request: Every generated file matches its generator's output (`tools/docgen: TestGeneratedDocsAreCurrent`)
- **X8** (new; target ci by M2) Every number records the exact versions and data it came from, and can be regenerated from them.
  - `new`, pull_request: No ledger unit is cited under two roles by any fit, gate or dashboard point
  - `new`, pull_request: Every data source is pinned to an exact revision
  - `new`, pull_request: Published numbers regenerate: on redistributable fixtures on every pull request, on the full data nightly
  - `new`, pull_request: An accuracy change's success criteria are committed before any commit that reads its held-out data
- **X10** (new; target ci by M1) Repositories are released bottom-up.
  - `new`, release: Release bot (§13.6) (`tools/release`)
- **X11** (partial; target ci by M1) Pins are exact release tags.
  - `ci`, pull_request: Selected-version lint (LIF-2), in every repository: no replace outside inference-sim (`tools/modlint: TestEachRuleHasAKnownBadInput`)
  - `ci`, pull_request: No other module requiring github.com/llm-d/llm-d-router (`tools/modlint: TestEachRuleHasAKnownBadInput`)
  - `ci`, pull_request: No committed go.work (`TestNoNestedModule`)
- **X12** (partial; target ci by M1) Only derived values are committed, never third-party data, with one exception: vendor config.json files whose licence is on the allow-list.
  - `new`, pull_request: Name scan in every repository, which also reads the text of every open issue each night
  - `ci`, pull_request: Provenance schema rules (`go run ./validate/cmd/validate-catalog catalog`)
  - `ci`, pull_request: A check that every committed config.json has an allow-listed licence in its model.yaml. It runs in CI on the project's own catalog; the loader does not require a licence record (`validate/cmd/validate-catalog: TestRunLicenceCheck`)
- **X13** (partial; target ci by M1) Every behaviour change has a test that fails without it.
  - `new`, pull_request: The corpus comparison, run by the scoring harness at the pinned digest from each repository's CI, for pull requests declared as refactors
  - `new`, pull_request: Landing gate (VAL-6)
  - `ci`, pull_request: Mutation check of the cost model's guarded properties: each is broken in turn and the test that guards it must fail (`ci.yml: mutants`)
  - `new`, pull_request: Mutation checks: on a pull request, the functions it changes and the functions covered by any test it changes are mutated, and only the tests that cover them run
  - `new`, nightly: The full sweep runs nightly and on release tags
- **X14** (new; target ci by M1) A pull request's description is accurate: the commit hashes it lists are the branch heads, it merges cleanly, and every invariant it cites exists.
  - `new`, pull_request: The description check, a shared workflow run in every repository (`shared-pr-checks.yml: prcheck description, invariants cited`)
- **X15** (ci; target reached) Each validation rule exists once, in Go, in blis-substrate.
  - `ci`, pull_request: The parity test in blis-substrate (`tools/pyrules: TestEveryRestatementAgreesWithGo`)
  - `ci`, pull_request: A lint that flags schema-like rule code in Python scripts (`tools/pyrules/pyrules.py`)
- **X16** (new; target ci by M1) A skipped test in a required or gate job is a failure, not a pass, unless it is on the quarantine list of X25.
  - `new`, pull_request: Each repository's CI configuration (`prcheck workflows, prcheck skips`)
  - `new`, nightly: The nightly status page (§13.4) (`shared-nightly.yml: prcheck status`)
- **X17** (partial; target ci by M1) Every repository's protected branch requires its named checks, including formatting and lint.
  - `ci`, practice: The protected branch requires the repository's named checks, including formatting and lint
- **X18** (new; target ci by M2) Every check in the invariant registry names a known-bad input that must make it fail.
  - `new`, nightly: A nightly job in blis-substrate that runs each check on its known-bad input and fails if the check passes (`nightly.yml: prcheck knownbad`)
  - `new`, nightly: The nightly mutation sweep (T7), which covers substrate validators and the simulator's own checks as well as engine code
- **X19** (new; target ci by M2) Scripts and generators, in every language, never write an engine default.
  - `new`, pull_request: The engine-name lint (X2), run on *.py and *.sh in every repository that has scripts
- **X20** (new; target ci by after M4) A commit written with an agent's help says so in one Assisted-by: <tool> line, and carries no Co-authored-by line naming the tool.
  - `new`, pull_request: A commit-policy check, required on every pull request in every repository, over the commits the pull request adds: one Assisted-by line, no Co-authored-by or Signed-off-by naming a tool (`shared-pr-checks.yml: prcheck commits`)
  - `new`, practice: The person who takes responsibility for a commit signs it and signs it off, or instructs an agent to push it in that person's name. Encouraged, never checked before the closing step after M4
- **X21** (new; target ci by M1) Everything a build or a release runs is pinned and traceable.
  - `new`, pull_request: A workflow lint that refuses an action not pinned by commit (`prcheck workflows`)
  - `new`, pull_request: Pip install --require-hashes in every Python CI job (`prcheck workflows`)
  - `new`, pull_request: A check that LICENSE and any required NOTICE exist
- **X22** (partial; target ci by M4) Every verdict (a CI result or the review agent's review, including its answers to review questions) names the git tree hash of the exact tree it judged.
  - `new`, pull_request: The shared pull-request check in blis-substrate/.github/workflows, called by every repository, which records each review-agent answer with the head tree
  - `new`, pull_request: A workflow lint that refuses ignored exit codes (`prcheck workflows`)
  - `ci`, pull_request: Each job's item count (`ci.yml job required`)
- **X23** (new; target ci by M1) A version is written in one place.
  - `new`, pull_request: A lint in every repository, over every file type, for version literals of BLIS modules or Go that are not generated (`prcheck versions`)
- **X24** (new; target ci by M4) Every review finding ends in one of three outcomes before merge: fixed, filed as a linked issue, or rejected with a reason.
  - `new`, pull_request: A pull-request check that refuses a review thread with no outcome (`shared-pr-checks.yml: prcheck threads`)
  - `new`, nightly: A nightly job that reopens an issue closed any other way
- **X25** (new; target ci by M1) Tests read only their declared inputs, never sibling checkouts, ignored directories or the user's environment.
  - `new`, pull_request: CI in a clean container with no sibling checkouts
  - `new`, pull_request: The quarantine list, which fails CI when an entry expires (`prcheck quarantine`)
  - `new`, pull_request: A required check that fails while main is failing and the pull request does not name the failing test's issue
- **X26** (new; target ci by M1) In documentation, every source path, symbol, issue or pull-request reference, and every statement that a feature is supported, is either generated (from surface.json, stack.json or the code) or checked against the pinned tree.
  - `new`, pull_request: A documentation reference check in every repository (`prcheck docrefs`)
- **X27** (new; target ci by M1) An open bug labelled as breaking an invariant blocks a release of the repository where the bug lives, and of each repository that ships that code, unless the release fixes only that bug or a security issue, or a release owner records a waiver in the release notes.
  - `new`, release: The release bot (`tools/release: CheckBugs`)
- **X28** (new; target ci by M1) An issue in a BLIS repository that is cited as the reason for a refusal, for a refused or not-modelled item in a field or surface list, or in a code comment that marks a gap or a divergence, stays open.
  - `new`, nightly: A nightly job in every repository that resolves each cited issue and fails on a closed one (`shared-nightly.yml: prcheck issues -resolve`)
  - `new`, pull_request: A pull-request check that refuses a change closing an issue still cited elsewhere (`shared-pr-checks.yml: prcheck issues -closes`)
  - `new`, nightly: The nightly job also reports an issue closed outside a merge
- **SUB-1** (partial; target ci by M2) Packages under contracts/ import only the Go standard library, the YAML parser and each other.
  - `ci`, pull_request: Import test (`contracts: TestKernelImportsOnlySpec`)
  - `new`, pull_request: API-compatibility check of the contracts/ packages against the last substrate tag
- **SUB-2** (ci; target reached) Every new or changed data field ships with its validator, a failing example for each way it can be wrong, and regenerated reference documentation.
  - `ci`, pull_request: Documentation-matches-types test (`contracts: TestDocFieldTablesMatchTypes`)
  - `ci`, pull_request: Strict loaders (`contracts: TestLoadRejectsUnknownFields`)
  - `ci`, pull_request: Examples must validate (`contracts: TestDocExamplesValidate`)
- **SUB-3** (partial; target ci by M4) The catalog holds declared facts only: figures read from a published source, each with a citation a reader can follow.
  - `ci`, pull_request: Deriver --check (`catalog/scripts/derive_graph.py --check`)
  - `new`, pull_request: The validators, including the packaging check and its test pair
  - `new`, pull_request: A test that no runtime package opens config.json
  - `new`, pull_request: A lint for model-name literals in kernel/ and in engine lowering code. Review: that each citation supports its figure
- **SUB-4** (partial; target ci by M1) Every coefficient is generated by its family's writer script, never edited by hand.
  - `new`, pull_request: Writers --check on fixtures per pull request, on full data nightly
  - `ci`, pull_request: Provenance rules (`go run ./validate/cmd/validate-registry registry`)
- **SUB-5** (partial; target ci by M2) Coefficients obey physics: no compute rate above the chip's dense datasheet peak for that data type; no memory bandwidth above the datasheet peak for a working set larger than the chip's L2 cache; at a fixed algorithm, topology tier and bytes per rank, a collective never gets faster as the group grows.
  - `ci`, pull_request: Property tests: no compute rate above the dense datasheet peak, and a collective never gets faster as the group grows (`registry/validator/test_coefficient_properties.py`)
  - `new`, pull_request: Property tests: the per-data-type peak and the L2-bounded memory bandwidth
- **SUB-6** (new; target ci by M4) The cost model resolves nothing.
  - `new`, pull_request: A test that the cost model's public functions accept only plans, step shapes, transfers, and catalog and registry handles
- **SUB-7** (new; target ci by M2) The cost model has a price for every primitive it can be asked about.
  - `new`, pull_request: Completeness test over every primitive and every catalog chip
- **SUB-8** (partial; target ci by M1) A change to the substrate never breaks the data it releases: validators, deriver and writers all pass on the tree being released.
  - `ci`, pull_request: The validators run on every pull request (`ci.yml jobs catalog, registry`)
  - `ci`, pull_request: The graph deriver --check runs on every pull request (`ci.yml job catalog: derive_graph.py --check`)
  - `new`, pull_request: Every registry writer --check runs on every pull request, on redistributable fixtures
- **SUB-9** (ci; target reached) Data is loaded the same way from a path or from the embedded copy: the same file patterns, the same validators (including the checks between the catalog and the coefficient registry), and the same content hash.
  - `ci`, pull_request: A test that loads the embedded copy and an exported checkout of the same commit, and gets identical data and an identical hash (`load: TestEmbeddedAndExportedCheckoutAreTheSame`)
  - `ci`, pull_request: A test comparing the two pattern lists (`TestEmbedPatternsEqualLoaderPatterns`)
- **SUB-10** (new; target ci by M2) contracts/conformance includes a set of unusual model shapes: every layer windowed, every layer recurrent, MLA only, a draft layer declared last, a single layer, and no attention at all.
  - `new`, pull_request: The conformance suite, run in blis-substrate, blis-vllm and blis-sglang
- **SUB-11** (new; target ci by M2) Every field of PoolPlan, StepShape and Transfer, and every coefficient family, changes some price when it is changed, or is listed as not yet priced with an issue.
  - `new`, pull_request: A perturbation test in kernel/
  - `new`, pull_request: A ratchet on the not-yet-priced list, which may only shrink
- **SUB-12** (partial; target ci by M2) Every loader of a BLIS document refuses unknown keys, duplicate keys, numbers that do not convert exactly, non-finite numbers and recursive aliases.
  - `ci`, pull_request: Each loader refuses unknown keys (`contracts: TestLoadRejectsUnknownFields`)
  - `new`, pull_request: A shared set of malformed documents in contracts/conformance, run against every loader
  - `new`, pull_request: A test that every declared field is read
- **SUB-13** (ci; target reached) blis-substrate is one Go module: no go.mod exists below the repository root, and the files in the module zip that Go fetches total under 100 MB uncompressed (Go's limit is 500 MB).
  - `ci`, pull_request: A check that fails on any nested go.mod (`TestNoNestedModule`)
  - `ci`, pull_request: A check of that uncompressed total (`TestModuleZipIsUnder100MB`)
- **SUB-14** (new; target ci by M2) The memory breakdown closes.
  - `new`, pull_request: In blis-substrate's error job, a comparison of the total with recorded GPU start-up logs, which are ledger units with role validate
- **SUB-15** (new; target ci by M4) A model graph holds only what the checkpoint states: the tensors, their shapes and stored data types, scale tensors, and the layer structure.
  - `new`, pull_request: A test that rederives every graph field from those inputs
  - `new`, pull_request: A check that graph vocabulary has no emit conditions. The deriver records each header's SHA-256 and the revision, under the same licence rule as config.json (X12)
- **SUB-16** (new; target ci by M2) Quantities carry units as types.
  - `new`, pull_request: The compiler
  - `new`, pull_request: A load-time unit check in the cost model
  - `new`, pull_request: Validator rules that catalog peaks are dense figures and that their ratios between data types match the datasheet. The bounds on rates are SUB-5's
- **SUB-17** (new; target ci by M3) A transfer or collective names its endpoints by rank, and the cost model prices it over the path between them through the catalog's links: the NVLink domain, the NIC and each switch tier.
  - `new`, pull_request: A property test over random placements, in blis-substrate
- **SUB-18** (new; target ci by M2) Provenance is recorded per field, not per file.
  - `new`, pull_request: Validator rules on catalog provenance fields
- **SUB-19** (partial; target ci by M1) Every document that crosses a repository boundary has one definition, in contracts/spec: the scenario, the deployment, the Result, and the per-point record the scoring harness writes.
  - `ci`, pull_request: A generator test in blis-substrate that the published schemas match the Go types (`tools/schemagen: TestPublishedSchemasMatchTheTypes`)
- **SUB-20** (new; target ci by M2) No price, size or engine decision depends on a free-text label.
  - `new`, pull_request: A syntax-tree lint in blis-substrate and each engine repository that refuses a comparison on a documentation-only string field outside tests
  - `new`, pull_request: A perturbation test (SUB-11) that changing such a label changes no price
- **DEV-8** (new; target ci by M2) Every reported metric names its definition and scope (§8.7), and a comparison is refused unless both sides have the same definition, scope and aggregate.
  - `new`, pull_request: The metric registry in contracts/, which the simulator reads directly and the dashboard and config search read through inference-sim metrics
- **DEV-12** (new; target ci by M1) Each change is a verification change or a validation change by what it does (§8.9), and is judged as such.
  - `new`, pull_request: In every repository, the change class in the pull-request template, and a check that the outputs that moved match the declared class (`.github/pull_request_template.md (generated)`)
- **VAL-1** (partial; target ci by M2) Every coefficient records exactly one method from the list in §9.2, or not_charged, with the sources that method needs: measured, calibrated: the data units it was fitted to; transferred: the source coefficient and the transfer rule; literature: an exact citation; vendor_spec: a datasheet citation; roofline: the catalog facts it was computed from; assumed, not_charged: a written reason.
  - `ci`, pull_request: Schema rules in blis-substrate/validate for the method vocabulary in use (`contracts/spec/coefficient: TestRationaleRequiredForEveryMethodButMeasured`)
  - `new`, pull_request: Schema rules in blis-substrate/validate for the method list of §9.2, with the sources each method needs
- **VAL-2** (new; target ci by M2) For every primitive family and every catalog chip, some rank has a value.
  - `new`, pull_request: Completeness test over every primitive and catalog chip (SUB-7)
- **VAL-3** (new; target ci by M2) Each price records, among the coefficients it used, the least trusted effective rank, and names its method and any flag.
  - `new`, pull_request: Step-time result field
  - `new`, pull_request: Checked in kernel tests
- **VAL-4** (new; target ci by M2) Each lookup resolves to one entry per rank, and the entry with the smallest rank number is used; flags do not change which entry is chosen, only how the price reports it.
  - `new`, pull_request: Validator check against the previous release, plus a review question
  - `new`, pull_request: A validator rule that refuses unordered matches
  - `new`, pull_request: A cost-model test that resolves every pair of scope kinds in both load orders and agrees with the validator (X15)
- **VAL-5** (new; target ci by M2) The typical error of each method is measured by leaving one chip out, and published per primitive family, using data units with role validate (§10).
  - `new`, nightly: Nightly job
- **VAL-6** (partial; target ci by M1) A change to a measured or calibrated value passes the landing gate.
  - `ci`, pull_request: On fixtures per pull request (`registry/validator (pytest)`)
  - `new`, nightly: On full data nightly
  - `new`, pull_request: Operator-level gates in blis-substrate
- **VAL-7** (new; target ci by M2) A coefficient scoped to a single model marks a cost that the cost model's general forms do not yet explain.
  - `new`, pull_request: Ratchet test
- **VAL-8** (new; target ci by M2) A lookup matches a coefficient only for the exact kernel implementation that runs.
  - `new`, pull_request: The cost model's lookup, which has no fallback to a sibling implementation
  - `new`, pull_request: A test that lists entries matching neither condition, which are removed in the next substrate release
- **VAL-9** (new; target ci by M2) A coefficient states what its measurement includes: device time only or host issue time as well, captured or eager execution, and which harness collected it.
  - `new`, pull_request: A validator rule matching each coefficient's measurement boundary with the boundary its reading formula declares
- **VAL-10** (new; target ci by M2) A measured or calibrated entry records the interval its fit supports (95%, by bootstrap or profile likelihood) and the range of inputs its fit data covered: shapes, token counts, ranks, chip and kernel.
  - `new`, pull_request: Validator rules on the interval and range fields
  - `new`, pull_request: A test that extrapolated lookups report it
  - `new`, pull_request: The fitting scripts' identifiability check
- **VAL-11** (new; target ci by M2) An operator-level check judges a pricing form; an end-to-end check judges how forms combine.
  - `new`, pull_request: A review question
  - `new`, pull_request: A lint that refuses comments or entries marked as a known over-charge or under-charge without a linked defect
- **D1** (partial; target ci by M2) Every value computed from ledger units cites them, whatever its method.
  - `ci`, pull_request: A validator rule (`go run ./validate/cmd/validate-registry registry`)
  - `new`, pull_request: The fitting scripts, which write the units they read
- **D2** (new; target ci by M1) No unit is read under two roles across all consumers.
  - `new`, nightly: A nightly join over every consumer's record of the units it read, run inside the nightly run of blis-substrate's gate job
- **D3** (ci; target reached) Every source revision is pinned by commit, hash or run id.
  - `ci`, pull_request: A ledger check that refuses latest, main and unknown as revisions (`ledger: TestUnpinnedRevisionsRefused`)
- **D4** (partial; target ci by M1) Roles are drawn by the keyed rule, per role key, so every unit sharing a role key has the same role, except a legacy unit, which keeps its legacy side even where units of its role key fall on both sides.
  - `ci`, pull_request: A check against the ledger at the pull request's merge base (`ledger: TestCheckAgainstBase`)
  - `ci`, pull_request: A check that no committed file lists roles, other than the three published lists that the release job writes and the legacy labels in legacy-map.yaml (`ledger: TestNoCommittedRoles`)
  - `new`, pull_request: A check that the committed unit list equals the ingest tool's output for the pinned source
  - `ci`, pull_request: A check that every unit at the merge base keeps its unit_id and role key, or is covered by an alias table or by a new partition version a person approved (`ledger: TestCheckAgainstBase`)
  - `new`, pull_request: A check that each alias table equals the one generated from the merge-base and candidate ingest tools on the same source revision
- **D5** (new; target ci by M2) A coefficient's fit data exercised the kernel implementation the coefficient is keyed by, meaning the same library, variant and kernel hash (CodeHash).
  - `new`, pull_request: A validator rule that a coefficient's fit units were collected on the kernel implementation it is keyed by
- **D6** (new; target ci by M4) Only redistributable artefacts are committed.
  - `new`, pull_request: A check that every committed data file is listed in the ledger as redistributable
  - `new`, nightly: A nightly job comparing fixtures with the full data
- **D7** (partial; target ci by M1) The role seed is 256 random bits and a CI secret until its partition version is retired.
  - `new`, pull_request: The seed is stored in one CI environment per repository (blis-substrate and accuracy-dashboard) whose branch policy allows only main, available only to the release, select, error and gate jobs
  - `ci`, pull_request: A workflow lint that those jobs check out only main (`ledger: TestLintWorkflows`)
  - `new`, pull_request: Their outputs are checked to contain no per-unit rows and no aggregate over fewer than 10 units
- **D8** (new; target ci by M2) A quantity BLIS simulates, such as prefix-cache hits or queueing, is never stored.
  - `new`, pull_request: A validator rule refusing such families in the registry without a workload key
  - `new`, pull_request: A test per borrowing step, for a tie, for each bound of K with and without stated per-position rates, for an ambiguous key, for the exact-only option and for the refusal
- **COV-1** (partial; target ci by M1) One scorer produces every number, and it lives only in accuracy-dashboard: no other repository compares a simulation with a measured end-to-end benchmark.
  - `new`, pull_request: A lint that refuses code outside accuracy-dashboard that reads end-to-end benchmark units to compute an error
- **LIF-1** (new; target ci by M1) A tag names one commit forever.
  - `new`, practice: Protected tags
  - `new`, nightly: A nightly check that every tag resolves to its recorded commit (`shared-nightly.yml: prcheck tags`)
- **LIF-2** (partial; target ci by M1) In every build, each BLIS module resolves to the exact version the release records in stack.json, and the router fork resolves to a -blis.N tag through inference-sim's one replace directive (X11).
  - `ci`, pull_request: Selected-version lint (go list -m all), which for the router reads the replace target that go list -m -json reports, not the placeholder version on the require line (`tools/modlint: TestEachRuleHasAKnownBadInput`)
  - `ci`, pull_request: A check that every module in go list -m -json all declares a GoVersion no higher than contracts/toolchain (`tools/modlint: TestEachRuleHasAKnownBadInput`)
  - `ci`, pull_request: A CI step that fails if go version differs from it (`tools/modlint: TestEachRuleHasAKnownBadInput`)
- **LIF-3** (new; target ci by M1) Only the release bot cuts releases, and only from a commit where every ci invariant passed.
  - `new`, practice: Repository settings that let only the release bot and people with write access create tags
  - `new`, pull_request: The bot's own check of every ci invariant (`tools/release: CheckInvariants`)
  - `new`, nightly: A nightly check that every tag made by a person is annotated and listed in its release notes
- **LIF-4** (new; target ci by M1) Release notes are generated (§13.3).
  - `new`, release: The release bot, which writes the notes from the release's changes (`tools/release: Notes`)
- **LIF-5** (new; target ci by M4) An engine module, or the router fork, models one upstream release.
  - `new`, release: A tag check in the release bot: format, N one above the previous tag on the same upstream release, no plain tag (`tools/release: CheckEngineTag`)
- **LIF-7** (new; target ci by M1) A deprecated feature is removed by the first release after its deprecation that may break compatibility: a MAJOR release, or a MINOR release while the repository is at v0.
  - `new`, release: The release bot, which refuses a MAJOR release, or a MINOR release at v0, that still contains a feature deprecated in an earlier release (`tools/release: CheckDeprecations`)

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

- **X1** Does this new scenario field describe something the real system would observe, rather than a setting the upstream does not have? Expertise: contracts expertise. Applies when: a key is added to the environment list. Paths: `contracts/spec/scenario/**`.
- **X2** Does this change hide one engine's policy (a default, threshold, backend choice or padding rule) behind a neutral name? Would the other engine need the same edit? Expertise: expertise in the changed directory. Applies when: any non-test change under catalog/, registry/ or kernel/. Paths: `catalog/**`, `registry/**`, `kernel/**`, except `**/*_test.go`, `**/testdata/**`, `**/tests/**`.
- **X6** Could this floating-point reduction give a different result on another platform? Expertise: expertise in the changed repository. Applies when: a parallel or floating-point reduction is added, or the cross-platform check fails. Added by a reviewer with the label `review:X6`.
- **X7** Does this change give a second home to a fact another repository owns? Expertise: expertise in the changed repository. Applies when: a pull request adds a literal list or table, or a constant that cites another repository. Added by a reviewer with the label `review:X7`.
- **X8** Was this fitting method, gate or threshold chosen after seeing evaluation results? Expertise: expertise in the repository landing the change. Applies when: a pull request changes a fitting method, gate or threshold. Paths: `registry/scripts/**`, `registry/validator/**`.
- **DEV-12** Does this change alter what is modelled? Expertise: expertise in the changed directory. Applies when: a pull request is declared a verification change but moves a Result outside the byte gate. Added by a reviewer with the label `review:DEV-12`.
- **VAL-11** Is this change kept or rejected because it offsets an error elsewhere? Expertise: expertise in the changed directory. Applies when: a pricing change lands or is rejected with end-to-end evidence that contradicts its operator-level evidence. Added by a reviewer with the label `review:VAL-11`.
- **D1** Was this value read from, or tuned against, any measurement? Expertise: registry expertise. Applies when: an assumed entry is added or changed. Paths: `registry/coefficients/**`.
- **D7** Was the new seed generated as the ledger records? Expertise: ledger expertise. Applies when: a seed is created or rotated. Added by a reviewer with the label `review:D7`.
<!-- END GENERATED: invariants -->
