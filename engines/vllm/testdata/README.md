# Test data for compat/kernelmodel

`scenarios/` holds inference-sim's own scenario fixtures (Scenario + Deployment documents written
for its tests), copied from inference-sim's `testdata/` at commit
`ca532abca209af41425c007974a5f3ef336f5981` when `compat/kernelmodel` (the latency kernel's
adapter, inference-sim#2021 B5) moved here. They are inputs to tests only.

The catalog and registry they resolve against are not here. They are the data of the
blis-substrate release `go.mod` pins, read through its loader: the embedded copy, or the
directories `BLIS_CATALOG` and `BLIS_REGISTRY` name (`compat/kernelmodel` `DefaultData`).
