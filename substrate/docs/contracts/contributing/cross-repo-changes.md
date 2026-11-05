# Changes across the BLIS repositories

<p class="lede">Most improvements to BLIS touch more than one of its six repositories. A schema
field, a catalog fact, a fitted coefficient, the kernel that reads them and the simulator that
calls the kernel have to land together, and every number published from them has to be
reproducible from what is public. This page is the one description of how such a change is
made. Each of the other five repositories links here and adds only its own steps.</p>

The rules below were worked out over several rounds of changes made across all the repositories
at once. They exist because each one, when skipped, produced a published number that nobody
could reproduce, a pull request that could not merge, or a fix that made the overall estimate
worse.

## The dependency order

| Repository | Depends on | How it pins what it depends on |
|---|---|---|
| **blis-schemas** | nothing in BLIS | — |
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | blis-schemas | CI runs `go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@<version> .` in `.github/workflows/ci.yml`; the README quotes the same version |
| [blis-registry](https://github.com/inference-sim/blis-registry) | blis-schemas, blis-catalog | `.github/workflows/validate.yml` checks out blis-schemas at a tag or a full commit; `.github/workflows/rederive.yml` pins the catalog (`CATALOG_SHA`) and the fitting data (`AISIMULATE_SHA`) by commit |
| [blis-vllm](https://github.com/inference-sim/blis-vllm) | blis-schemas | `go.mod` (and `rules/go.mod`, its rules-pack module) requires blis-schemas |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | blis-schemas; blis-vllm's rules packs in its commands and tests only; catalog and registry as test inputs | `go.mod` requires blis-schemas and blis-vllm/rules; `testdata/upstream.lock` names a catalog and a registry commit, which `scripts/fetch-testdata.sh` fetches and verifies |
| [inference-sim](https://github.com/inference-sim/inference-sim) | blis-latency-kernel, blis-vllm, blis-schemas; catalog and registry as vendored test data | `go.mod` requires the kernel, blis-vllm and blis-schemas; `testdata/catalog` and `testdata/registry` are vendored copies with `.upstream-commit` stamps; `testdata/upstream-prerelease.lock` overlays the kernel's lock while the inputs are unreleased |

Changes flow one way: schemas, then catalog, registry and blis-vllm, then the kernel, then
inference-sim. Nothing
imports in the other direction. blis-schemas' own CI reads a registry checkout at `main` to
check that committed data still parses; that job does not test an open registry pull request.

### Go pseudo-versions

A Go consumer pins an unreleased commit by its pseudo-version,
`v<next patch>-0.<UTC commit time>-<first 12 hex digits of the SHA>`, for example
`v0.3.1-0.20261017092034-afd839aa6053`. Once the commit is pushed,
`GOPROXY=direct go list -m github.com/inference-sim/blis-schemas@<sha>` prints it. The `go.sum`
hash is computed from the module's content, so it is the same before and after the push.

A pseudo-version pin is a pre-release state. Before a consumer merges, the pin moves to a
release tag of the dependency. A comment beside every pin of an unreleased commit says
`TEMPORARY` and names the pull request whose head it is, so a reviewer cannot mistake it for
a release.

### No `replace` in anything pushed

A `replace` directive in `go.mod` points a module at a local path. It builds on one machine and
nowhere else, and a published number computed with it cannot be reproduced. To develop
against an unpushed dependency, use an uncommitted `go.work` file, or the `insteadOf` mapping
of the dependency's URL to a local clone, and keep both out of the commit. Before every commit
that touches `go.mod`:

```sh
grep -n '^replace' go.mod   # must print nothing
go mod tidy && git diff --exit-code go.mod go.sum
```

When a branch is merged into an integration branch and it carries a `replace`, the merge drops
the `replace` and pins the dependency's pushed head instead.

## Order of pushes and merges

Every pull request in a cross-repository change stays a draft until all of them are ready. Then:

1. **Push blis-schemas.** Its head is the pin every other repository uses.
2. **Repin and push the middle four**, in any order except that blis-vllm comes before the
   kernel, whose tests use its rules packs:
    - blis-catalog: the `validate-catalog@<version>` pin in `ci.yml` and the README;
    - blis-vllm: `go.mod` and `rules/go.mod`;
    - blis-registry: the schema ref in `validate.yml`, and `CATALOG_SHA` in `rederive.yml` if
      the change needs a newer catalog;
    - blis-latency-kernel: `go.mod` (blis-schemas and blis-vllm/rules), and `testdata/upstream.lock` if the tests need the new
      catalog or registry; regenerate `docs/generated/` (its `sources.md` lists the pin).
3. **Repin, re-vendor and push inference-sim**: `go.mod` to the kernel, blis-vllm and schemas heads;
   `testdata/catalog` and `testdata/registry` from the catalog and registry heads, with their
   `.upstream-commit` stamps; `testdata/upstream-prerelease.lock` to those heads.
4. **Run every repository's suite against the others' pushed heads**, not against local
   checkouts.
5. **Regenerate every pull request body** from the pushed heads (see [Parity](#parity)).

If any head changes after a downstream repository pinned it, repeat the steps below it. A
change to the schemas head means repinning all five.

Merging follows the same order, with a release between each step: merge blis-schemas and tag
it; move the catalog's and registry's CI pins to that tag and merge each, then tag them; move
the kernel's `go.mod` and `upstream.lock` to the tags, merge and tag it; move inference-sim's
`go.mod` to the kernel and schemas tags, re-vendor the catalog and registry from their tags,
empty `upstream-prerelease.lock`, and merge. inference-sim's
`TestTheVendoredCatalogAndRegistryAreTheLockedReleases` fails while the prerelease lock
restates the kernel's lock, and `TestDocExamplesPinTheCatalogVersion` fails while its
documentation names an older catalog release.

## Keeping every pull request mergeable

Before every push, in each of the five branches:

```sh
git fetch origin
git merge origin/main                               # resolve, then re-run the suite
git merge-tree --write-tree origin/main HEAD        # must exit 0 and print one tree line
```

A non-zero exit, or conflict lines after the tree, means GitHub will report a conflict. A merge
that conflicts is resolved in a commit whose message lists each file, both sides, and what was
kept. An integration merge that breaks a test neither branch broke on its own ("integration
breakage") is fixed in its own commit, named as such.

## Parity

**Every published number must be reproducible from the public pull request heads alone.** That
means:

- no coefficient that exists only in a local snapshot or an uncommitted file;
- no `replace` (above);
- the registry and catalog vendored into inference-sim equal the registry and catalog pull
  request heads, file for file;
- every CI pin and every `go.mod` pin equals the head of the pull request it names;
- every pull request body names the exact head SHA of its own branch and of each dependency
  it pins, and those SHAs are the branch heads at the moment of the push.

A push gate checks the last point mechanically: every commit SHA quoted in a body must equal
the current head of the branch it names. A body written before the final merges goes stale, so
bodies are regenerated from the heads as the last step before a push, never edited by hand.

A published run names the five commits it was built from. A reader reproduces it by checking
out those commits and building, without anything from the author's machine. Tools and
harnesses that produce a published number belong in a pull request; external data they read
is named by URL and revision or sha256, not committed.

## Integration and gates

### One binary from the integrated heads

All five branches are merged and pinned to each other first. The simulator is then built once,
from a detached checkout of the integrated inference-sim head, with its revision injected:

```sh
go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRevision=$(git rev-parse HEAD)" -o blis main.go
shasum -a 256 blis
go version -m blis | grep inference-sim/blis-   # the kernel and schemas versions it contains
```

Every gate runs on that binary. Its sha256, its revision and the module versions it contains
are recorded with the results.

### Pinned, immutable snapshots

The registry and catalog a run reads are snapshots, not working trees. A snapshot is
`git archive <sha>` of the pinned commit, extracted to a directory named `<repo>@<sha12>` and
made read-only. Before a run, a pin check refuses a root that is inside a git working tree,
whose name does not match the pinned commit, that has a writable file, or whose per-file sha256
manifest differs from the commit's. A working tree with an uncommitted edit once fed
unpinned coefficients to validation runs; the snapshot rule exists because of it.

### Gates

A gate is written down, with its threshold and its baseline build, **before** the data it judges
exists, and is never relaxed after the data is seen. The kinds used so far:

| Kind | Asks | Example |
|---|---|---|
| Transfer | Is the aggregate error on a fixed set of measured configurations, not used for fitting, no worse than the baseline's? | output-throughput and interactivity MAPE on another part's measured runs, under fixed limits |
| Ordering | Where more than one configuration was measured, does BLIS order them the same way? | per-level order agreement, ties broken by magnitude |
| Anchor | On request-level measured runs, are solo and load-matched ITL error, TTFT error and the KV-pool ratio no worse than on the baseline build? | per-anchor rows, each with a no-worse verdict |
| Startup and levers | Does the build refuse what the engine refuses, and does each knob still move the quantity it governs? | an engine whose KV cannot hold one request is refused; each lever test is mutation-checked |

**The landing rule.** A fix lands only if it improves, or does not worsen, its held-out anchors.
Before and after figures come from a script whose output path is quoted, never typed by hand.
A fix that fails its gate does not land. The exception is a recorded deviation (below), decided
by a maintainer and stated openly in the pull request body with its evidence.

**Seeds.** A latency comparison between two builds uses the mean over the same seeds (for
example 42 to 45) on both. "No worse" means
`mean(new) - mean(old) <= sqrt(var(old)/n(old) + var(new)/n(new))`, with sample variances over
seeds. A single seed is not evidence for or against a change.

**The same points.** A held-out comparison scores both builds on the same points: the same
point identifiers, the same request set, the same metric window. A subsample is drawn by a
seeded, stratified rule, and its file and sha256 are recorded before either side is scored. A
hit rate is compared on the requests both builds complete, not over a window whose request mix
shifts with latency.

**Apples to apples.** Each measured run is replayed as that run was configured: the engine
behaviour of the release it actually ran, its client settings, its metric definitions. A
setting BLIS cannot yet express is a named mismatch, listed, and the affected metric is not
scored. A comparison between two different quantities is not made at all.

**Coefficient and pricing changes** follow blis-registry's
[fitting principles](https://github.com/inference-sim/blis-substrate/blob/main/docs/registry/fitting-principles.md):
a declared fit / held-out split written before the held-out numbers are read, distributions
rather than point errors, agreement across latency, KV, queueing and throughput together, the
version rule, and the landing gate there.

## Version discipline

BLIS models one engine release at a time. A statement about what the engine does cites the
release's source as `<path>:<lines>@<tag>`. Data measured on another engine version is used only
for a term whose code path is unchanged between that version and the modelled release, shown by
a per-function diff that is recorded with the fit; otherwise the term is fitted only from data
on the modelled release, or parked. A schema change that extends a frozen client or engine lock
is recorded as a deliberate version extension.

## Tests

Every behaviour change has a test that fails without it. Check that by mutation: revert the fix
(or break the law) on purpose, see the test fail, restore it, and say in the commit message what
was broken and which test caught it. Tests assert behaviour through public interfaces, one
behaviour each, with the smallest setup that triggers it. A test that skips because its data is
absent prints why; an unexpected skip is a test that did not run.

## Public artifacts

Commits, code, comments, documentation, issue and pull request text are public.

- **Never write internal project, customer or colleague names in a public artifact.** Refer to
  private measurements by what they are ("internal deployment measurements"), not by where they
  came from. A name scan of the diff, the commit messages and every tracked file runs before
  each push. A private-names check (`.github/workflows/private-names.yml`) reads its pattern
  from a repository secret and fails a pull request whose files, paths, commit messages, title
  or body match it.
- **Never redistribute third-party data.** Measurements, traces and benchmark dumps stay where
  their owner publishes them. A repository commits derived values only, each with its
  provenance: the source URL and its commit or sha256.

## Review before each push

1. **An expert self-audit.** Before an integration point and before every push, write down the
   ten correctness questions each of three experts would ask of all the pull requests together:
   an engineer of the modelled engine, a benchmark engineer who produced the measurements, and a
   distributed-inference simulation researcher. Answer each from code and data, with file and
   line evidence, and grade it sound, weak or wrong. Fix every wrong answer that affects
   correctness before pushing; file the rest. This extends inference-sim's
   [pre-commit self-audit](https://github.com/inference-sim/inference-sim/blob/main/docs/contributing/pr-workflow.md#step-475-pre-commit-self-audit)
   and its [review perspectives](https://github.com/inference-sim/inference-sim/blob/main/docs/contributing/perspectives.md)
   to every repository.
2. **A holistic review across all the pull requests**, as one change, not one per repository, with the
   findings that affect correctness fixed before the push.

## Recording decisions and deviations

So that the next round does not re-derive them, every decision that is not obvious from the code
is written down when it is made:

- **pre-registrations**, with their time, before the data they govern: partitions and their
  hashes, gates and thresholds, subsample rules, the baseline build;
- **landing decisions**: what landed, what did not, and the gate result that decided it;
- **deviations**: a gate that failed but whose change landed anyway, with who decided, why,
  and the evidence (seed spread, per-row figures);
- **remaining gaps**, each as a drafted issue with a title and body.

The decisions that bind a pull request appear in its body, under its evidence. A pull request
body says plainly when a gate did not improve.

## Before a push: the checklist

- [ ] Every branch has merged `origin/main`; `git merge-tree --write-tree origin/main HEAD` is clean.
- [ ] No `replace` in any `go.mod`; every pin equals the head it names.
- [ ] The vendored catalog and registry in inference-sim equal those heads.
- [ ] Every repository's suite passes against the others' pushed heads.
- [ ] Gates were run on one binary built from the integrated heads, with pinned snapshots.
- [ ] Behaviour changes have mutation-checked tests, and held-out evidence from a script.
- [ ] The self-audit and the cross-repository review are done; correctness findings are fixed.
- [ ] The name scan is clean; no third-party data is committed.
- [ ] Pull request bodies are regenerated from the heads, and every SHA in them is a branch head.
