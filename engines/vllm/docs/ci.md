# Continuous integration

How `.github/workflows/ci.yml` decides what to run, how it shares the slow suites over runners, and how to
reproduce each decision on a laptop. The design is two ideas: run a job only when a changed file is one the job
reads, and cut the jobs that must run into pieces that run side by side, each piece proving it ran what it was
given.

## Contents

1. [The checks and when each runs](#the-checks-and-when-each-runs)
2. [How a change selects jobs](#how-a-change-selects-jobs)
3. [Required checks and skipped jobs](#required-checks-and-skipped-jobs)
4. [Sharing the Go tests](#sharing-the-go-tests)
5. [The race detector](#the-race-detector)
6. [The oracle jobs](#the-oracle-jobs)
7. [Caches](#caches)
8. [Reproduce it locally](#reproduce-it-locally)
9. [Costs and limits](#costs-and-limits)

## The checks and when each runs

On a pull request the `changes` job lists the files the pull request changes and the jobs below run only
when it names a file they read. On a push to main and on a manual run (`gh workflow run ci.yml --ref <branch>`)
every job runs. The last column says what a job reads: the files whose change selects it.

| Job | Checks | Runs when a changed file is |
|---|---|---|
| `go-build` | gofmt, no `replace`, `go vet`, `go build` | a `.go` file, `go.mod`, `go.sum`, or a file a package embeds |
| `go-test-rest` (3 shards) | every package except `kerneltests`, without `-race` | in a package of the module outside `kerneltests` (sources, tests, testdata), or data those tests open from elsewhere (`testdata/`, `kerneltests/testdata/`, `vllmengine/testdata/`, `oracle/scenarios*/`, `oracle/mocks.json`, `oracle/drafts/`, `fields.json`, `docs/metrics-*.json`) |
| `go-test-kerneltests` | `./kerneltests/...` | in `kerneltests/` outside `_inpackage/` and `scripts/`, or in a package its tests link (sources only), or `go.mod`/`go.sum` |
| `go-test-inpackage` | the kernel tests that run inside blis-substrate's kernel package | in `kerneltests/_inpackage/`, `kerneltests/testdata/`, `kerneltests/scripts/`, a package those tests link, or `go.mod`/`go.sum` (the pinned substrate) |
| `go-race-guard` | every package with a goroutine, channel, atomic, mutex or Once is raced; the legs are the guarded packages | a non-test `.go` file, or the legs file |
| `go-race` (8 legs) | the race detector over the packages that hold concurrency, and over the census test | a leg runs when its packages, what they link, or the data they read changed |
| `go-bench` | the speed benchmarks of `core/sched` and `core/vllm` (recorded, not thresholded) | in those packages or a package their tests link, or `go.mod`/`go.sum` |
| `lint` | golangci-lint v2.14.0 | a `.go` file outside `testdata/` and `_` directories, `go.mod`, `go.sum`, `.golangci.yml` |
| `packs-against-vllm` (3 names) | each rules pack against the vLLM release it transcribes | the pack's package (`rules/v0_29`, `rules/v0_31`, `rules/v0_32`), a package its tests link, or `go.mod`/`go.sum` |
| `metrics-manifest` | `fields.json` and the metrics manifest are current for vLLM v0.32.0; registry entries are connectors vLLM registers | `fields/`, `fields.json`, `docs/metrics-manifest.json`, `cmd/metricsmanifest`, `registry/` and what they link |
| `goldens-reviewed` | a commit that changes a golden names its reviewer; the oracle's guards fail on what they guard | `oracle/`, a `testdata` directory under `core/`, or `tools/migrateengineblock/` |
| `oracle-scenarios`, `oracle-inertness` (3 shards), `oracle-fuzz` (6 corpora) | vLLM's own output against the committed goldens, the inertness proofs, and the fuzz corpora, and blis-vllm against them | `oracle/`, `.github/actions/oracle-fuzz/`, `docs/metrics-*.json`, the data the oracle's steps name (`core/kv/testdata/`, `core/lower/testdata/`, `compat/quantization/resolved.json`, `rules/engineconfig/testdata/`, `rules/v0_32/testdata/`), or a package the oracle's Go tests link |

Every job runs when `.github/workflows/ci.yml`, the selection (`changed_jobs.py`, `audit_triggers.py`,
`test-changed-jobs.sh`) or `.github/actions/go-setup/` changes, because they change what runs. The rest of
`.github/` is read by the jobs that use it and by no other: the shard tool (`shard.py`, `run-go-shard.sh`,
`go-test-items.sh`, `go-test-json.sh`) and the weights and split list run the Go test jobs (`shard.py` and
`weights-inertness.txt` also run the inertness shards); `race-legs.json` and
`check_race_legs.py` run the race legs and the guard; `.github/actions/oracle-fuzz/` and `oracle-env/` run the
oracle jobs. The other workflows (`blis.yml`, `mutation.yml`, `oracle-traces.yml`, `private-names.yml`), the
pull-request template and scripts that only people run (`measure_weights.py`) run no job of `ci.yml`.
A data file under `.github/` therefore never counts as shared, which would run the whole workflow for one
changed weight.

Documents only (`docs/*.md`, `README.md`, `AGENTS.md`, `LICENSE`, `NOTICE`) select nothing: the cheap jobs that
always report (`changes`, `ci-self-test`, and the aggregates) are all that run.

## How a change selects jobs

`.github/scripts/changed_jobs.py` reads the changed paths on standard input and prints `job=true|false`.

- **Go tests.** A job that runs Go tests is selected by a file in the directory of a package it tests (the
  package's tests and testdata are its input), by a source or embedded file of a package those tests link
  (`go list -deps -test` names them, so a new import changes the selection with no edit), and by the data its
  tests open from outside their package. A package that is only linked contributes its sources, not its tests
  or testdata: a golden under `core/sched/testdata/` does not run `go-test-kerneltests`, which links
  `core/sched` but reads no golden.
- **Everything else** is a list of paths in the script, one list per job.
- **The nearest package owns a file.** A file belongs to the deepest package directory above it, so a change
  to `kerneltests/_inpackage/` does not select the tests of `kerneltests/`.

The lists are checked against the repository, not against memory. `audit_triggers.py` finds every path a Go
test opens from outside its own package, every repository path a step of `ci.yml` names, and every repository
path a Python file under `oracle/` names, and fails unless changing that path selects the job that reads it.
`test-changed-jobs.sh` holds sample changes with the exact set of jobs each selects (a document, `fields.json`,
an oracle golden, a rules file, a kernelmodel file, a workflow, `go.mod`), runs the audit, and runs the tests of
the shard tool. The `ci-self-test` job runs it on every change, so a new file read that the lists do not know
fails the build instead of skipping a job. To see what a change would select without a pull request:

    printf '%s\n' compat/kernelmodel/acceptance.go | python3 -I .github/scripts/changed_jobs.py

or run the workflow by hand with the `changed` input: `gh workflow run ci.yml --ref <branch> -f changed="docs/testing.md"`.

## Required checks and skipped jobs

Branch protection on main requires `go`, `lint`, `oracle-tier2`, the three `packs-against-vllm (...)` names,
`metrics-manifest` and `goldens-reviewed`. A required name has to be reported by every run, or the merge waits
for it. So:

- `go` and `oracle-tier2` are aggregates. They run on every change, list the jobs they cover in `needs`, and
  pass only if each of those succeeded or was skipped by the selection. A job that failed or was cancelled
  fails the aggregate. `changes` and `ci-self-test` are in `go`'s `needs`, so a failed selection fails `go`.
- `lint`, `metrics-manifest`, `goldens-reviewed` and `packs-against-vllm` always start. When nothing selected
  them, their steps are skipped and the job passes in a few seconds. (A matrix job that is skipped as a whole is
  reported under its unexpanded name, which would leave the three `packs-against-vllm (...)` names waiting.)
- Everything else (the shards, the legs, the oracle parts) is not a required name and is skipped by `if:`.

## Sharing the Go tests

`go-test-rest` runs three shards. The tests of the group are listed (`go-test-items.sh`): one item for each
package, and, for the packages named in `.github/shards/split.txt` (`compat/kernelmodel`, `vllmengine`,
`compat/build`, whose tests alone take longer than a shard), one item for each top-level test. Every shard lists
the whole group itself and takes its share with `shard.py select`:

1. items sorted by measured weight, heaviest first (`.github/shards/weights.txt`, seconds from one `go test
   -json` run; an item with no weight weighs 1 second, so a new test lands in a shard and is never dropped);
2. each item goes to the shard with the least weight so far (ties by shard index).

The shards are therefore the classes of one function of the item list. A package or test that is in no shard is
not possible, and `run-go-shard.sh` checks the three ways it could still happen:

- `shard.py check` recomputes the whole partition and fails unless every item is in exactly one shard, no shard
  is empty, and the list is not suspiciously short (a broken `go list` would otherwise hide tests from every
  shard);
- `shard.py verify` reads the shard's `go test -json` and fails unless every item of its share has a pass or
  skip result and none failed;
- `.github/scripts/test_shard.py` tests the partition on random lists and the failure messages.

A shard's test names are uploaded as the artifact `tests-rest-<k>` (`package`, `test`, `result`, one line each,
subtests included), so the executed set can be compared with any other run.

To add a package, do nothing: it is listed and placed. To split a package that has become slow, add it to
`split.txt`. To re-balance after a large change, regenerate the weights:

    go test -count=1 -json ./... > /tmp/results.json
    python3 -I .github/scripts/measure_weights.py /tmp/results.json > .github/shards/weights.txt

## The race detector

`.github/shards/race-legs.json` lists the legs. A leg is a group of packages (`core`, `compat`), a shard of a
package whose tests are shared out by name as above (`vllmengine-0..1`), or a slice of the census test
(`kernelmodel-census-0..3`). `check_race_legs.py`, run by `go-race-guard`, fails unless the legs' packages are
exactly the guarded `RACE_PKGS` of `ci.yml`, the shard legs of a package are shards 0..n-1 once each, and the
slices are k/n for k in 0..n-1 once each.

The census test (`TestBlockSizeResolutionOverTheCensusFixtures`) took 13 minutes under `-race` alone, because it
opens 679 fixtures as parallel subtests. `BLIS_CENSUS_SLICE=k/n` runs the fixtures at the positions k, k+n,
k+2n, ... of the sorted list, so four legs together run every fixture once (`TestCensusSlicesPartitionTheFixtures`
tests the rule). A slice cannot assert the census's exact totals, since it sees part of the fixtures; the
unsliced run in `go-test-rest` asserts them, so no assertion is dropped. Each slice still asserts the per-fixture
facts (a refusal cites its issue; a differing stated size is listed).

## The oracle jobs

The former `oracle-tier2` job is `oracle-scenarios` (the scenario-driven oracles and the goldens),
`oracle-inertness` (the tests of `oracle/test_inertness.py`, shared out by name over three jobs with the same shard
tool, listed by unittest so a test added later lands in a shard; each job proves it ran its share by the count
unittest reports) and
`oracle-fuzz` (one job for each of the six fuzz corpora: the corpus from the cache or generated, and the Go
tests that compare blis-vllm with it), all side by side. `oracle-tier2` is their aggregate. Both build the same
pinned environment, `.github/actions/oracle-env`. A corpus is a pure function of its generator, seeds, runners,
mock, lock and the vLLM commit, so `.github/actions/oracle-fuzz` keys it on exactly those; a miss regenerates it.
The cache of main is read by every pull request; a pull request that changes a generator or a runner misses once.

## Caches

`.github/actions/go-setup` installs the Go that `go.mod` names and restores two caches. setup-go's own cache is
off: its key was the runner image, Go and `go.sum` for every job, so the first job to finish saved it and the others
failed to reserve it ("another job may be creating this cache"), and an image change (ubuntu24 to ubuntu26)
emptied it. Now the module cache is keyed on `go.sum`, and each kind of job has its own build cache keyed on the job
group, `go.sum` and the commit, restoring the newest earlier entry of the group. Go's caches are
content-addressed, so a stale entry is never wrong, only less useful. Pushes to main and manual runs save them; pull
requests only read them, so they never evict main's entries. The pinned Python packages use setup-python's pip
cache, keyed on `oracle/requirements-hashed.txt`.

## Reproduce it locally

    # with the Go that go.mod names on the PATH
    .github/scripts/test-changed-jobs.sh                       # the selection's own tests and the audit
    .github/scripts/run-go-shard.sh -exclude '/kerneltests' -- rest 3 1 ./...   # shard 1 of the rest
    BLIS_CENSUS_SLICE=2/4 go test -race -run TestBlockSizeResolutionOverTheCensusFixtures ./compat/kernelmodel

## Costs and limits

Time on a hosted runner (4 cores), queueing excluded, measured on 2026-11-13 on the tree of the combined pull
request with the fuzz caches warm. A job's time is checkout, Go and caches (about 25 seconds) plus its work. The
"before" column is the median of eleven runs of the same tree before this change.

| Job | Before | After |
|---|---|---|
| `go-race (kernelmodel-census)` | 771 s | 4 legs of 190 to 210 s |
| `go-race (vllmengine-a)` and `(vllmengine-b)` | 121 s and 343 s | 2 legs of 170 to 300 s |
| `go-test-rest` | 320 s | 3 shards of 76 to 143 s |
| `go-test-kerneltests` (with the in-package tests) | 297 s | 108 s and 229 s, side by side |
| `oracle-tier2` | 2065 s (1184 s of it generating corpora on a cache miss) | `oracle-scenarios` 284 s; `oracle-inertness` 3 shards of 256 to 344 s; `oracle-fuzz` 6 jobs of 105 to 162 s (about 770 s for the slowest on a miss) |
| `lint`, `metrics-manifest`, `packs-against-vllm` | 101 s, 193 s, 30 s | 30 s, 187 s, 30 s |

The wall-clock of a run is the slowest of these plus the time its jobs wait for a runner, which depends on what else
the organisation is running; a run of the whole workflow is the sum of about 39 jobs. Runner minutes rise
(each piece repeats its setup) while the time to a result falls.

Limits:

- GitHub runs a limited number of jobs of an organisation at once. A run of this workflow is about 39 jobs, so when
  several runs are active the cheap jobs queue behind the slow ones.
- The selection is only as complete as its lists. The audit covers the reads it can see (string paths in tests, in
  `ci.yml` and in the oracle's Python); a test that builds a path at run time from a variable it never states as a
  literal escapes it. The mitigation is that every Go job is also selected by any change to a package it links, and
  by a change to `ci.yml` or the selection.
- A pull request's checks judge its whole diff against the base each time; nothing is carried over from an earlier push.
- Weights are seconds on one developer machine, not on a runner; only their ratios matter.
