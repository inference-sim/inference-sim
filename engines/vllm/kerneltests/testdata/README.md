# Test inputs

What this directory holds, and where each part comes from.

| path | what | in git? | where it comes from |
|---|---|---|---|
| `*.yaml` | hand-written scenario+deployment fixtures | yes | this repository |
| `aisimulate/`, `direct/` | generated scenario+deployment fixtures | yes | `scripts/kernel/gen_aisimulate_scenarios.py`, `scripts/kernel/gen_direct_*.py` |

The catalog and registry these fixtures resolve against are not here. They are
`substrate/catalog/` and `substrate/registry/`, read through `internal/substratedata`;
`BLIS_CATALOG` and `BLIS_REGISTRY` override them. A pinned
copy that fails to load fails the test (`internal/artifacttest`); only an override may skip.

