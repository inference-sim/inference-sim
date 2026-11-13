# How CI decides what to run

`.github/workflows/ci.yml` runs the build, the linter and the tests. A pull request runs only the
jobs its changed files need, the slow packages run as parallel shards, and a Go build cache saved
from `main` is restored by every job. This page states the rules, the evidence for them, and how
to check that nothing was dropped.

## Contents

- [The rule](#the-rule)
- [Which change runs which job](#which-change-runs-which-job)
- [Shards](#shards)
- [Required checks](#required-checks)
- [The build cache](#the-build-cache)
- [Previewing and forcing](#previewing-and-forcing)
- [What this does not do](#what-this-does-not-do)

## The rule

The `changes` job lists the files the pull request changed (the merge commit against its first
parent) and `scripts/ci-changes.sh` turns the list into sixteen yes/no groups, one per job. The
rules are fail-closed:

- A path that no class in the script names turns every group on.
- A push to `main`, a tag, and every `workflow_dispatch` without `changed_paths` turn every group
  on. The delivery loop dispatches `ci.yml` and reads a green run as "the whole suite passed".
- An empty change list turns every group on.

A class names the jobs that read a path. Go packages follow the import graph: a file in the
directory of a package a job tests turns that job on; a non-test `.go` file of a package those
tests import does too; a `_test.go` file of an imported package does not, because no other package
compiles it. `scripts/ci_changes_test.go` re-derives the graph with `go list -deps -test` for each
matrix entry of `ci.yml`, and for the blis binary (the byte gate builds it, and selected-deps links
it, embedded files included), so a new import cannot leave a job off.

The byte gate runs when a non-test Go file of a package the blis binary links, a file it embeds
(`defaults.yaml`), or the gate's own inputs change. The selected-deps job runs on the same Go
files, because an added import changes which blis-vllm and router packages the binary links and
so which packages it tests, and on `go.mod`, `go.sum` and `stack.json`. A change to a `_test.go`
file, to `tools/`, or to a document runs neither.

Data files follow what the tests open. Each tree was deleted on a scratch branch and the whole
suite run: only the jobs listed below failed. The same classes are in `ci-changes.sh`.

| Path | Jobs it turns on |
|------|------------------|
| `docs/**`, `README.md`, `CONTRIBUTING.md`, `mkdocs.yml` | core, cluster, cmd, scripts, stack |
| `testdata/bytegate/**` | byte gate (both architectures), workload, cmd |
| `testdata/kernel/**` | cmd, core, cluster, kernel backend, other packages, scripts |
| `testdata/trace-v3/**` | workload |
| `examples/**`, `examplescenarios/**` | workload, cmd |
| `stack.json`, `guides.lock` | cmd, other packages, stack, selected-deps, root |
| `.github/ISSUE_TEMPLATE/**`, the pull request template, `LICENSE`, `specs/**` | none |
| `.github/workflows/<other>.yml` | scripts, actionlint |
| `scripts/ci-*.sh`, `ci.yml`, `go.mod`, `go.sum` | everything |

Only workflow files under `.github/` are shared inputs. A data file there is not.

## Shards

`./cmd` (about 410 s of tests), `./sim/workload`, `./sim/llmd` and the selected-deps packages are
split so the longest job is two minutes, not ten. A test belongs to shard `cksum(name) mod N`,
computed from `go test -list`, so the shards partition the tests by construction, a new test lands
in one shard with no list to edit, and an added test never moves another (a test checks this).

Two checks keep it honest. Each shard fails if the tests it started differ from the tests it owns.
The job after the shards, `scripts/ci-shard-verify.sh`, fails unless the shards' lists are
identical, their shares are pairwise disjoint, their union is the package's full list, and the
number of shards is the number the workflow declares (a shard whose evidence never arrived is an
error). Selected-deps splits blis-vllm's `compat/kernelmodel` by test (it is most of that job) and
every other package by import path.

To change a shard count, edit the matrix and the number passed to `ci-shard-verify.sh` together.

## Required checks

`main` requires `build`, `lint` and five `test (...)` checks. They keep their names and always
report: a job whose group is off skips its steps and succeeds, because a job skipped under a
matrix template never reports the name the ruleset waits for. The two required names whose
packages are sharded (`test (workload & cmd, ./sim/workload ./cmd, 10m)`) are jobs that wait for
the shards and fail if any shard failed, was cancelled, or the evidence check failed.

The job `required` is the single check to require if the ruleset is ever simplified: it succeeds
when every job succeeded or was skipped and the `changes` job succeeded. It is not required today.

## The build cache

Each job restores `~/.cache/go-build` saved by the `cache-warm` job, which runs on every push to
`main` (and on a dispatch with `save_cache`). The key holds the OS, the architecture, the hash of
`go.mod` and `go.sum` and the commit; a restore takes the newest entry with the same dependencies,
else the newest of any. Pull requests never save, since one entry per push would fill the
repository's 10 GB quota. The module cache is not cached: tests start with an empty one (X25), and
the download is a few seconds. `golangci-lint` is built from source with the module's Go and the
binary is cached per Go version.

## Previewing and forcing

```bash
# What would a pull request touching these paths run? (dispatch on any branch)
gh workflow run ci.yml --ref <branch> -f changed_paths=$'sim/cluster/instance.go\ndocs/index.md'

# Everything, as the delivery loop does
gh workflow run ci.yml --ref <branch>

# Everything, and save this branch's build cache
gh workflow run ci.yml --ref <branch> -f save_cache=true
```

Locally, `git diff --name-only --no-renames origin/main... | scripts/ci-changes.sh` prints the
groups.

## What this does not do

- The filter reads file paths, not what a test calls. A new test that opens a file outside the
  classes above is not seen until the class is added; the failure mode is a missing job on the
  pull request and the job running on `main`. Add the class when a test starts reading a new tree.
- Skipping a job on a pull request does not skip it on `main`: every push to `main` runs all.
- The workflows other than `ci.yml` (`gofmt`, `private-names`, `blis`, `docs`) are unchanged.
