# Vendored BLIS artifacts

A pinned copy of the upstream artifacts the kernel-path tests and scoring commands read:
scenario fixtures from [blis-latency-kernel], hardware and model facts from [blis-catalog],
fitted coefficients from [blis-registry], and the AISimulate/InferenceX measurement
corpora.

[blis-latency-kernel]: https://github.com/inference-sim/blis-latency-kernel
[blis-catalog]: https://github.com/inference-sim/blis-catalog
[blis-registry]: https://github.com/inference-sim/blis-registry

`sim/kernelmodel/roots.go` resolves against this tree by default.

## Provenance

| tree | upstream | commit |
|---|---|---|
| `scenarios` | blis-latency-kernel `testdata/aisimulate` | `545ac11b7954934e8add404ffcb586803e6f7777` (2026-10-08) |
| `measurements` | blis-latency-kernel `testdata/measurements` | `545ac11b7954934e8add404ffcb586803e6f7777` (2026-10-08) |
| `catalog` | blis-catalog | `747a2213e13030ae675f9872c3e2a76b6b770d50` (2026-10-07) |
| `registry` | blis-registry | `46d930c0fb96e603b76c507acaf14910d48c4ccb` (2026-10-07) |

`545ac11` is the blis-latency-kernel commit this module pins in `go.mod`, so the fixtures
and the kernel that prices them come from one revision. The `catalog` and `registry`
commits are the ones blis-latency-kernel itself vendors, so all four repositories agree:
`registry` here is byte-identical to the kernel's pinned copy, and `catalog` differs from
it only by the nine `config.json` files noted below.

Copied verbatim -- no edits.

## What is here, and why only this much

| path | count | what |
|---|---|---|
| `scenarios/*.yaml` | 88 | scenario+deployment fixtures (schemas v0.2.0) |
| `catalog/hardware/*.yaml` | 9 | chip descriptors |
| `catalog/networks/*.yaml` | 3 | inter-node fabrics |
| `catalog/devices/storage.yaml` | 1 | storage-device facts |
| `catalog/models/<name>/graph.yaml` | 32 | derived model graphs |
| `catalog/models/<name>/config.json` | 9 | vendor HF configs, for the roofline arm |
| `registry/coefficients/*.yaml` | 5 | fitted coefficient sets |
| `measurements/*.json` | 5 | AISimulate and InferenceX corpora |

2.7M rather than the ~28M the four repositories occupy, because of two deliberate cuts.

**Model `config.json` is vendored for nine models, not all thirty-two.** blis-latency-kernel
vendors none: it reads only the derived `graph.yaml`, and the configs are the vendor
provenance that graph was derived *from*. This repository additionally runs a roofline
estimator arm that parses the HF config directly, so the configs those scenarios name are
required -- `TestAnalyticArmsChangeStepTimeNotAdmission` fails without them. Nine is the
exact set the 88 scenarios reference. The excluded twenty-three include
`nemotron-3-ultra-550b-a55b-nvfp4` (6.6M) and `nemotron-3.5-lightning-30b-a3b-nvfp4`
(1.3M), whose configs carry per-tensor quantization maps; no vendored scenario names
either, so vendoring them would add 7.9M -- three times this whole tree -- for nothing.

**Five measurement corpora, not eleven.** The other six (`inferencex_direct*.json`,
`corpus.json`, 5.1M) are consumed by blis-latency-kernel's own `cmd/direct`; nothing in
this repository references them in Go, docs, or scripts. Scoring against them stays
possible -- the corpus flags take a path, and `BLIS_MEASUREMENTS` redirects the lot.

## Why vendored rather than read from a sibling checkout

The tests read twenty-six absolute paths under one developer's home directory
(`/Users/sri/Documents/Projects/...`) across eleven files, and every affected test turned a
failed read into a SKIP. Twenty-three such skip-on-missing sites existed in
`sim/kernelmodel` alone. A skip reads as a pass, so the suite was both unrunnable for
anyone else and silent about it.

That is not hypothetical in either direction:

- **blis-latency-kernel hit it.** Under the pseudo-version it was pinned to, the sibling
  catalog had already moved to blis-schemas v0.2.0 field names (`read_bandwidth_mb_s`) that
  the pinned schema rejected. Every test building a kernel from a committed fixture skipped
  on `catalog unavailable` -- 41 of them, with the suite still reporting `ok`.
- **This repository hit the mirror image.** The sibling blis-latency-kernel checkout sat on
  a branch carrying pre-v0.2.0 scenario fixtures (`hardware` and `pools` at the scenario top
  level, before the `Scenario`/`Deployment` split). Under schemas v0.2.0 every scenario read
  failed, and `./sim/kernelmodel/...` went red with eight failures that said nothing about
  this repository's own code.

Against this tree the same suite is 65 pass, 0 skip, 0 fail. The pinning is what buys that:
a coefficient or chip rename is *supposed* to break the fixtures that name it, which is why
the copy tracks a commit rather than a branch.

## Overrides

Each root is independently redirectable, so a working copy can be scored against live
upstream checkouts without editing anything:

```
BLIS_SCENARIOS     scenario+deployment fixtures
BLIS_CATALOG       blis-catalog root
BLIS_REGISTRY      blis-registry root
BLIS_MEASUREMENTS  measurement corpora
```

A bad override fails loudly rather than skipping -- verified both ways: a nonexistent
`BLIS_CATALOG` fails the smoke test with the missing path named, and pointing the pair at
live sibling checkouts passes.

## Updating

Re-copy the paths in the table above from upstream checkouts, update the commit table, and
run `go test ./...`.

Note `.gitignore`: the repository blanket-ignores `*.json` with an explicit allowlist, so
`measurements/*.json` and `catalog/models/**/config.json` are enumerated there. A new
corpus added here without an allowlist entry is silently untracked. The binary patterns
`blis`, `blis_iter*` and `inference-sim` were also unanchored, so a bare `blis` matched any
path component of that name and excluded this entire directory from `git add` -- they are
now `/blis`, `/blis_iter*`, `/inference-sim`.
