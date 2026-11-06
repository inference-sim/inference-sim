# blis-substrate

blis-substrate is the part of BLIS that is true whatever engine runs. It has four parts, each
a section of this site:

| Section | Directory | What it holds |
|---|---|---|
| [Contracts](contracts/index.md) | `contracts/` | the interfaces and document types every BLIS repository implements or calls, their validators, and the conformance suites |
| [Catalog](catalog/index.md) | `catalog/` | published facts about hardware, networks, storage, workloads and models, each model's graph derived from its vendor configuration |
| [Registry](registry/index.md) | `registry/` | fitted coefficients, each with its method, sources and data units |
| [Cost model](kernel/index.md) | `kernel/` | the Go code that prices an engine step from the catalog and the registry |

One repository holds all four because they change together: a model with something new in
its architecture usually needs a new data field, a catalog entry, new coefficients and a
change to the cost model at once. In one repository that is one pull request, one review and
one release.

The engine repositories and the router fork import only the packages under `contracts/`.
Only inference-sim imports the cost model and the data.

A run of the simulator uses the catalog and registry embedded in the substrate release it
was built with. `--catalog` and `--registry` name directories that replace the embedded
copy, and `--catalog-overlay` and `--registry-overlay` add entries to it. Every run records
which source it used and the content hash of each, so a result can be traced to the exact
data behind it. [Point BLIS at the catalog](catalog/using/point-blis.md) gives the details.

A change that spans several of these parts is one pull request. [Changes across
repositories](contributing/changes.md) describes such changes, and changes that reach other
BLIS repositories. [Documentation and releases](contributing/documentation.md) describes how
this site is built and how a release is published.
