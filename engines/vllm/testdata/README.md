# Test data for compat/kernelmodel

`compat/kernelmodel` (the latency kernel's adapter, moved from inference-sim, #2021 B5) is tested
against the same inputs it was tested against in inference-sim. These directories are copies of
inference-sim's `testdata/` at commit `ca532abca209af41425c007974a5f3ef336f5981`:

- `catalog/`: blis-catalog at the commit in `catalog/.upstream-commit`
  (`676326ec6c708e91ec4981bd6b03922a7fe60512`, the 0.3.0 release);
- `registry/`: blis-registry at the commit in `registry/.upstream-commit`
  (`7f6662d7188dc3dc873a37623e1f7bbfe1d6081b`, the v0.2.0 release);
- `scenarios/`: inference-sim's own scenario fixtures (blis-schemas Scenario + Deployment
  documents written for its tests).

They are inputs to tests only. No coefficient or model value here is new: each file is
byte-identical to its source. `compat/kernelmodel` points at this directory through
`SetVendoredRoot` in its tests (`main_test.go`). A simulator that builds the adapter
points it at its own vendored copies instead.
