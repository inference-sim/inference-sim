---
hide:
  - navigation
---

# blis-schemas

<p class="lede">BLIS, the Blackbox Inference Simulator, estimates how an LLM serving
deployment will perform before anyone runs it. blis-schemas defines the documents BLIS
reasons about and checks them. It is a Go module: types for a model, its hardware, the
coefficients a cost model uses, a deployment, its traffic and a measured run; a validator
for each; the interface a cost model implements; and the contract between a simulator and the
request router it hosts. It holds no data.</p>

## The idea in brief

A document's *shape* is defined here, and its *contents* live in the repository that owns
them, so either side can change without the other. A **scenario** fixes the problem: the
model, the hardware, the traffic, and the engine version. A **deployment** states the choices
made against it: parallelism, engine settings, offload. Validation asks two questions and
keeps the answers apart. Is each document well formed? That never depends on an engine
release. Would the release the scenario names run it as written? That always does. Every
validator reports every problem at once, each naming the field that is wrong.

--8<-- "docs/contracts/figures/repositories.html"

## Where to start

<div class="grid cards" markdown>

-   **You edit catalog or registry data**

    ---

    Run the validators the data repositories run in CI, and look up what a key means.

    [Validate a catalog or registry](guides/checkouts.md) ·
    [Reference](reference/index.md)

-   **You write Go that reads these documents**

    ---

    Load documents, validate them together, and read the report.

    [Validate documents from Go](guides/go.md) ·
    [Kernel interface](reference/kernel.md) ·
    [Router contract](reference/embed.md)

-   **You want to know why it is built this way**

    ---

    The ideas the schemas rest on, then the reasoning behind particular choices.

    [Concepts](concepts/index.md) ·
    [Design notes](design/index.md)

-   **You are changing blis-schemas**

    ---

    Where a change belongs, how an engine release is added, how these pages are built and
    published.

    [Contributing](contributing/index.md)

</div>

## Install

```sh
go get github.com/inference-sim/blis-schemas@latest
```

The two command-line validators run without a checkout of this repository:

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@latest  /path/to/blis-catalog
go run github.com/inference-sim/blis-schemas/cmd/validate-registry@latest /path/to/blis-registry
```

These pages are versioned. The selector in the header switches between release lines;
`dev` follows the main branch.
