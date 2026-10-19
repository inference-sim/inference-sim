# BLIS Roadmap: llm-d Parity by Design

> Where BLIS (Blackbox Inference Simulator) is going: the fast inner loop for evolving llm-d
> without spending GPUs on every iteration. We plan to bring BLIS to
> [llm-d-perf-simulator](https://github.com/llm-d-incubation/llm-d-perf-simulator) as the
> community-maintained **llm-d performance simulator**.

**Guiding principle: llm-d parity by design.** BLIS mirrors vLLM + llm-d behavior, models only the
load-bearing parts (*selective fidelity*), and tracks platform drift.

**Ecosystem split.** Responsibility is factored across repos so each evolves on its own cadence:

- **inference-sim**: the DES engine (workloads, scheduling, KV, batching, placement).
- **blis-catalog**: declared facts, i.e. hardware datasheets, model spec (possibly derived), fabrics.
- **blis-registry**: learned latency coefficients, each carrying its provenance and scope.
- **blis-schemas**: the shared file formats and their validators.
- **blis-latency-kernel**: step-time, memory, and transfer pricing physics.
- **llm-d-router**: the real EPP routing physics, embedded via a DES-compatible fork.
- **accuracy-dashboard**: nightly scoring of BLIS vs real latency.
- **leaderboard** / **blis-config-search**: config search and comparison on top of BLIS.

## 1. llm-d parity

BLIS runs the real **llm-d-router** inside the simulator via a fork patched only for determinism,
ensuring parity with llm-d routing physics. Port the remaining unsupported EPP plugins; publish a
matrix of what BLIS supports versus llm-d.

## 2. Day-0 model support

Land a new model's declared facts in **blis-catalog** so it runs immediately, with its latencies as
estimates until validated. Then make it *scored* (accuracy-validated) through a fast
coefficient-onboarding path into **blis-registry**, aided by architecture-family priors.

## 3. Multi-node accuracy

Price pipeline parallelism (`pp > 1`); model concurrent P/D handoff contention (#1922); support
multi-node replay; extend validation to larger topologies and new GPU families.

## 4. Networking configs

First-class `NetworkConfig` keyed by fabric not GPU; activate hop-latency and contention-aware
transfer pricing; add new fabric classes as catalog contributions.

## 5. Accuracy transparency

Maintain and extend **accuracy-dashboard** to demonstrate BLIS's accuracy trajectory against a
variety of publicly available ground-truth data sources, such as InferenceX.

## 6. Config search

Search lives in **leaderboard** / **blis-config-search**; BLIS exposes a stable programmatic run
interface and sweep + score primitives, and aims to rank candidate configs in the same order the
real cluster would.
