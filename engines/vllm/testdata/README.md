# Test data for compat/kernelmodel

`compat/kernelmodel` (the latency kernel's adapter, moved from inference-sim, #2021 B5) is tested
against the same inputs it was tested against in inference-sim. These directories are copies of
inference-sim's `testdata/` at commit `ca532abca209af41425c007974a5f3ef336f5981`:

- `catalog/`: a subset of blis-substrate's `catalog/` at the release `go.mod` pins, byte for
  byte (blis-substrate copied blis-catalog 0.5.0 unchanged);
- `registry/`: blis-substrate's `registry/coefficients/` at that release, byte for byte
  (blis-registry v0.3.0);
- `scenarios/`: inference-sim's own scenario fixtures (Scenario + Deployment documents written
  for its tests).

`catalog/` and `registry/` are removed when blis-substrate's loader (`load/`) is released, and
`compat/kernelmodel` reads the substrate's embedded data instead.

They are inputs to tests only. No coefficient or model value here is new: each file is
byte-identical to its source. `compat/kernelmodel` points at this directory through
`SetVendoredRoot` in its tests (`main_test.go`). A simulator that builds the adapter
points it at its own vendored copies instead.
