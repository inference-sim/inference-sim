# Test inputs

What this directory holds, and where each part comes from.

| path | what | in git? | where it comes from |
|---|---|---|---|
| `*.yaml` | hand-written scenario+deployment fixtures | yes | this repository |
| `aisimulate/`, `direct/` | generated scenario+deployment fixtures | yes | blis-latency-kernel's `scripts/gen_aisimulate_scenarios.py`, `scripts/gen_direct_*.py` |

The catalog and registry these fixtures resolve against are not here. They are
blis-substrate's `catalog/` and `registry/` at the release `go.mod` pins, read from the module
cache (`internal/substratedata`); `BLIS_CATALOG` and `BLIS_REGISTRY` override them. A pinned
copy that fails to load fails the test (`internal/artifacttest`); only an override may skip.

They used to be read from a sibling checkout under one developer's home directory, and a failed
read became a SKIP. Under the pseudo-version this repository was pinned to, the sibling catalog
had already moved to blis-schemas v0.2.0 field names the pinned schema rejected, so every test
that built a kernel from a fixture skipped -- 41 of them, 46 counting subtests -- with the suite
reporting `ok`.
