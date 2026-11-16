# blis-sglang

A Go model of the SGLang inference engine's scheduling, KV-cache, hierarchical-cache,
disaggregation and metrics behaviour, for the BLIS simulator. One build models one SGLang
release, named in `provenance.go`; its decisions are checked against that release's own Python
classes, run on CPU with only the GPU, the clock, transports and the platform replaced
(`oracle/`).

The engine port is not written yet. This repository holds its skeleton: the release pin, the
shared checks from blis-substrate's `contracts/harness`, and CI. `AGENTS.md` gives the rules for
a change.
