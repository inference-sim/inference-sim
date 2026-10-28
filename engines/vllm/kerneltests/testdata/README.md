# Test inputs

What this directory holds, where each part comes from, and why some of it is not committed.

| path | what | in git? | where it comes from |
|---|---|---|---|
| `*.yaml` | hand-written scenario+deployment fixtures | yes | this repository |
| `aisimulate/`, `direct/` | generated scenario+deployment fixtures | yes | `scripts/gen_aisimulate_scenarios.py`, `scripts/gen_direct_*.py` |
| `upstream.lock` | the upstream commits the next two rows are fetched at | yes | this repository |
| `catalog/` | chip, fabric and storage descriptors; derived model graphs | **no** | [blis-catalog], fetched |
| `registry/` | fitted coefficient sets | **no** | [blis-registry], fetched |
| `measurements/` | measured latencies the scoring commands compare against | **no** | third-party publications; see below |

[blis-catalog]: https://github.com/inference-sim/blis-catalog
[blis-registry]: https://github.com/inference-sim/blis-registry

## Running the tests

```
scripts/fetch-testdata.sh      # once, and again whenever upstream.lock changes
go test ./...
```

`scripts/fetch-testdata.sh` fetches each upstream at the commit `upstream.lock` pins, checks the
fetched HEAD against it, copies only the paths the lock lists (chip, fabric and storage
descriptors and model graphs from the catalog; coefficient sets from the registry), and stamps
each directory with `.upstream-commit` and an `.upstream-manifest` of each copied file's SHA-256.
It is idempotent: a directory whose stamp names the locked commit and whose files all match the
manifest is left alone, and anything else is fetched again. `TestTheFetchedUpstreamsAreTheLockedCommits`
fails if a stamp does not match the lock or a file does not match the manifest, so neither a
lock bump without a re-fetch nor a damaged copy can pass silently. A damaged copy is not
hypothetical: a `git pull` that crosses the commit which stopped tracking these directories
deletes the files it used to track, and leaves the stamp behind.

To read a live upstream checkout instead, set `BLIS_CATALOG` / `BLIS_REGISTRY` for the tests, or
pass `-catalog` / `-registry` to the commands.

## Why fetched rather than committed, and why a missing copy fails

The catalog and registry are other repositories' artifacts. Committing a copy here made three
copies of each (upstream, here, and in inference-sim) that had to be re-copied by hand on every
release; the lock makes the pin one line.

They used to be read from a sibling checkout under one developer's home directory, and a failed
read became a SKIP. Under the pseudo-version this repository was pinned to, the sibling catalog
had already moved to blis-schemas v0.2.0 field names the pinned schema rejected, so every test
that built a kernel from a fixture skipped -- 41 of them, 46 counting subtests -- with the suite
reporting `ok`. So the pinned copy is never optional: an absent `catalog/` or `registry/` fails
with the command to run (`internal/artifacttest`), and only an explicit override may skip.

## Measurements, and the `scoring` suite

`measurements/` is NOT distributed with this repository. Its corpora are measured latencies
extracted from third-party publications -- SemiAnalysis InferenceX's database dump and NVIDIA
AISimulate's end-to-end accuracy artifact -- by the `scripts/extract_*` tools, whose docstrings
name each source and the release they were pinned to. The scenarios generated from them carry
only deployment configurations, and are committed.

`cmd/score` and `cmd/shape` read the corpora from `BLIS_MEASUREMENTS`, or from `measurements/`
here, and say so when they are missing. The tests that need them are behind the `scoring` build
tag:

```
BLIS_MEASUREMENTS=/path/to/corpora go test -tags scoring ./cmd/...
```

Opting in makes a missing corpus a failure, not a skip. The default `go test ./...` -- and CI --
runs everything else. So do `scripts/gen_aisimulate_scenarios.py --check` and
`scripts/test_direct_corpus.py`, which read the corpora too.
