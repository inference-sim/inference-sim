# Test data for core/kernelmodel

`scenarios/` holds the scenario fixtures (Scenario + Deployment documents) written for the
kernel-model tests. They are inputs to tests only.

The catalog and registry they resolve against are not here. They are the data of
`substrate/catalog/` and `substrate/registry/`, read through its loader: the embedded copy, or the
directories `BLIS_CATALOG` and `BLIS_REGISTRY` name (`core/kernelmodel` `DefaultData`).
