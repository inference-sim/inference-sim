# Changes across repositories

<p class="lede">A change that touches several parts of blis-substrate is one pull request.
A change that also touches another BLIS repository is a review set: one pull request in each
repository, reviewed and released as one change. This page describes both, the evidence a
change that moves a published number carries, and the rules for public text.</p>

## Within blis-substrate

The contracts, the catalog, the registry and the cost model are directories of one
repository and one Go module. A new data field, its validator, the catalog entries that use
it, new coefficients and the cost model change that reads them land in one pull request.
CI runs the job for each directory the change touches and the checks between directories,
all at the same commit: the validators against the data, and the cost model against the data
it is released with (design §3.4). There is nothing to pin between them.

Before opening the pull request, run the checks for every directory it touches:

```sh
gofmt -l . && go vet ./... && go test ./...
go run ./validate/cmd/validate-catalog catalog
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check && python3 -m pytest tests -q)
(cd registry && python3 -m pytest validator -q)
DOCS_RELEASE=main python3 -m pytest docs/hooks -q
DOCS_RELEASE=main mkdocs build --strict
```

## Across BLIS repositories

The other repositories depend on blis-substrate in one direction (design §2.1):

| Repository | Imports from blis-substrate |
|---|---|
| blis-vllm, blis-sglang, the llm-d router fork | only the packages under `contracts/` |
| inference-sim | the whole module: the contracts, the cost model, the data loader and the embedded data |
| blis-config-search, accuracy-dashboard | nothing; they use inference-sim's commands and the JSON Schema each substrate release publishes |

A coefficient or pricing change therefore needs a substrate release and an inference-sim
release, and no engine or router release. A contract change reaches every repository that
implements the contract.

A change that spans repositories is one review set. One file, `stack.lock`, lists each
repository's branch head. During review, stack CI builds and tests the heads together,
through a `go.work` file it generates and never commits. Each repository keeps its released
pins until the set is approved. The release bot then merges and tags each repository bottom
up, blis-substrate first, writes each pin once, and runs the next layer's checks against the
new tags (X10, design §13.6).

Pins are exact release tags. No module commits a `replace` directive, except inference-sim's
one for the router fork, and no repository commits a `go.work` file (X11). To develop
against an unpushed dependency, use an uncommitted `go.work` file and keep it out of every
commit.

An incompatible change to a package under `contracts/` is made only as a contract-change
review set, which releases every implementer against it together (SUB-1).

## Evidence for a change that moves a number

### Gates

A gate is written down, with its threshold and its baseline build, **before** the data it
judges exists, and is never relaxed after the data is seen (X8). Operator-level gates need
only the cost model and the registry, and run in blis-substrate. Whole-run gates need
engines and the simulator, and run accuracy-dashboard's scoring harness, the only end-to-end
scorer, at the digest the repository pins (design §11.7, §13.5). Each tree's catalog and
registry are passed to the simulator with `--catalog` and `--registry`.

**The landing rule.** A fix lands only if it improves, or does not worsen, its held-out
units. Before and after figures come from the harness or a committed script, never typed by
hand. A fix that fails its gate does not land, unless a defect elsewhere explains the result
and a person approves it, as VAL-11 allows; the pull request says so with its evidence.

**Seeds.** A latency comparison between two builds uses the mean over the same seeds on
both. "No worse" means
`mean(new) - mean(old) <= sqrt(var(old)/n(old) + var(new)/n(new))`, with sample variances
over seeds. A single seed is not evidence for or against a change.

**The same points.** A held-out comparison scores both builds on the same points: the same
units, the same request set, the same metric window. A hit rate is compared on the requests
both builds complete, not over a window whose request mix shifts with latency.

**Like with like.** Each measured run is replayed as it was configured: the engine release
it ran, its client settings, its metric definitions. A setting BLIS cannot express is a named
mismatch, and the affected metric is not scored.

**Coefficient and pricing changes** follow the registry's
[fitting principles](../registry/fitting-principles.md): a fit and held-out split declared
before the held-out numbers are read, distributions rather than point errors, agreement
across latency, KV, queueing and throughput together, the version rule, and the landing gate.

### Recorded data

Every Result records the source and content hash of the catalog and the registry it used,
the hash of each overlay, and whether the run used released data (design §3.3). A figure
computed on a branch is therefore labelled experimental, and can be posted on a pull request
but never published. A published number comes from released data only (X8).

### Version discipline

BLIS models one engine release at a time. A statement about what the engine does cites the
release's source as `<path>:<lines>@<tag>`. Data measured on another engine version is used
only for a term whose code path is unchanged between that version and the modelled release,
shown by a per-function diff recorded with the fit; otherwise the term is fitted only from
data on the modelled release, or parked.

## Tests

Every behaviour change has a test that fails without it (X13). Check that by mutation: break
the fix on purpose, see the test fail, restore it, and say in the commit message what was
broken and which test caught it. Tests assert behaviour through public interfaces, one
behaviour each, with the smallest setup that triggers it. A test that skips because its data
is absent prints why; an unexpected skip is a test that did not run.

## Public text

Commits, code, comments, documentation, issue and pull request text are public.

- **No private names.** Never write internal project, customer or colleague names in a
  public artifact. Refer to private measurements by what they are ("internal deployment
  measurements"). A private-names check (`.github/workflows/private-names.yml`) reads its
  pattern from a repository secret and fails a pull request whose files, paths, commit
  messages, title or body match it (X12).
- **No third-party data.** Measurements, traces and benchmark dumps stay where their owner
  publishes them. The repository commits derived values only, each with its provenance: the
  source URL and its commit or SHA-256 (X12). Vendor `config.json` files whose licence is on
  the allow-list are the one exception.

## Review before a push

1. **An expert self-audit.** Write down the correctness questions each of three experts
   would ask of the change: an engineer of the modelled engine, a benchmark engineer who
   produced the measurements, and a distributed-inference simulation researcher. Answer each
   from code and data, with file and line evidence, and grade it sound, weak or wrong. Fix
   every wrong answer that affects correctness before pushing; file the rest.
2. **One review across a review set**, as one change, not one per repository.

## Recording decisions

Every decision that is not obvious from the code is written down when it is made:

- **pre-registrations**, with their time, before the data they govern: gates and thresholds,
  subsample rules, the baseline build;
- **landing decisions**: what landed, what did not, and the gate result that decided it;
- **deviations**: a gate that failed but whose change landed anyway, with who decided, why,
  and the evidence;
- **remaining gaps**, each as an issue with a title and body.

The decisions that bind a pull request appear in its body, under its evidence. A pull request
body says plainly when a gate did not improve.
