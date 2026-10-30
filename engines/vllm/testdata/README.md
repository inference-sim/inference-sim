# Test data for compat/kernelmodel

`compat/kernelmodel` (the latency kernel's adapter, moved from inference-sim, #2021 B5) is tested
against the same inputs it was tested against in inference-sim. These directories are copies of
inference-sim's `testdata/` at commit `ca532abca209af41425c007974a5f3ef336f5981`:

- `catalog/`: blis-catalog at the commit in `catalog/.upstream-commit` (TEMPORARY: the pushed
  head of inference-sim/blis-catalog#47, until it is released);
- `registry/`: blis-registry at the commit in `registry/.upstream-commit` (TEMPORARY: the
  pushed head of inference-sim/blis-registry#62, until it is released);
- `scenarios/`: inference-sim's own scenario fixtures (blis-schemas Scenario + Deployment
  documents written for its tests).

They are inputs to tests only. No coefficient or model value here is new: each file is
byte-identical to its source. `compat/kernelmodel` points at this directory through
`SetVendoredRoot` in its tests (`main_test.go`). A simulator that builds the adapter
points it at its own vendored copies instead.
