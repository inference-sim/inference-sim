# Test inputs

What this directory holds, where each part comes from, and why some of it is not committed.

| path | what | in git? | where it comes from |
|---|---|---|---|
| `*.yaml` | hand-written scenario+deployment fixtures | yes | this repository |
| `aisimulate/`, `direct/` | generated scenario+deployment fixtures | yes | blis-latency-kernel's `scripts/gen_aisimulate_scenarios.py`, `scripts/gen_direct_*.py` |
| `measurements/` | measured latencies the scoring commands compare against | **no** | third-party publications; see below |

The catalog and registry these fixtures resolve against are not here. They are
blis-substrate's `catalog/` and `registry/` at the release `go.mod` pins, read from the module
cache (`internal/substratedata`); `BLIS_CATALOG` and `BLIS_REGISTRY` override them. A pinned
copy that fails to load fails the test (`internal/artifacttest`); only an override may skip.

They used to be read from a sibling checkout under one developer's home directory, and a failed
read became a SKIP. Under the pseudo-version this repository was pinned to, the sibling catalog
had already moved to blis-schemas v0.2.0 field names the pinned schema rejected, so every test
that built a kernel from a fixture skipped -- 41 of them, 46 counting subtests -- with the suite
reporting `ok`.

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
