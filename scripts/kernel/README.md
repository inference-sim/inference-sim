# The cost model's fixture generators

These scripts write and check the scenario fixtures under `testdata/kernel/`. They came from
blis-latency-kernel v0.5.0 (commit 560bda63998700b1745c84387008ce3c1ded1b2a, `scripts/`) with
the fixtures (release plan §2).

| script | writes |
|---|---|
| `gen_aisimulate_scenarios.py` | `testdata/kernel/aisimulate/` from `aisimulate_e2e.json`, and `testdata/kernel/direct/` from `inferencex_direct.json` |
| `extract_aisimulate_e2e.py`, `extract_inferencex_*.py` | the corpora, from the third-party publications their docstrings name |
| `gen_direct_corpus.py`, `gen_direct_absolutes.py`, `gen_direct_engine_settings.py` | the direct corpus's scoring inputs |
| `priceprobe/` | one line of step prices per fixture, for the price-change report |

The corpora are not committed in any BLIS repository; they are read from `BLIS_MEASUREMENTS`.
To check that the committed fixtures are current:

    python3 -I scripts/kernel/gen_aisimulate_scenarios.py "$BLIS_MEASUREMENTS/aisimulate_e2e.json" -o testdata/kernel/aisimulate --check
    python3 -I scripts/kernel/gen_aisimulate_scenarios.py "$BLIS_MEASUREMENTS/inferencex_direct.json" -o testdata/kernel/direct --check

The tests that need no corpus run anywhere:

    python3 -I scripts/kernel/test_gen_aisimulate_scenarios.py
    python3 -I scripts/kernel/test_extract_inferencex_engine_settings.py

`internal/kernelfixtures` holds the Go shape test of every fixture (the scenario and its
deployment, two documents, one name, pools that fit the cluster).
