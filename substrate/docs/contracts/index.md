---
hide:
  - navigation
---

# Contracts

<p class="lede">BLIS, the Blackbox Inference Simulator, estimates how an LLM serving
deployment will perform before anyone runs it. The contracts, the packages under
<code>contracts/</code> in blis-substrate, define the documents BLIS reasons about and check
them: types for a model, its hardware, the coefficients a cost model uses, a deployment, its
traffic and a measured run; a validator for each; the interface a cost model implements; the
interface an engine implements; and the contract between a simulator and the request router
it hosts. They hold no data. Every Go repository in BLIS imports them.</p>

## The idea in brief

A document's *shape* is defined here. Its *contents* live in the catalog, in the registry,
or in the scenario and deployment a user writes. A **scenario** fixes the problem: the
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

    Run the validators CI runs on `catalog/` and `registry/`, and look up what a key means.

    [Validate the catalog and the registry](guides/checkouts.md) ·
    [Reference](reference/index.md)

-   **You write Go that reads these documents**

    ---

    Load documents, validate them together, and read the report.

    [Validate documents from Go](guides/go.md) ·
    [Kernel interface](reference/kernel.md) ·
    [Router contract](reference/embed.md)

-   **You want to know why it is built this way**

    ---

    The ideas the contracts rest on, then the reasoning behind particular choices.

    [Concepts](concepts/index.md) ·
    [Design notes](design/index.md)

-   **You are changing the contracts**

    ---

    Where a change belongs, how it is tested, and the rule that contracts change only
    compatibly.

    [Contributing](contributing/index.md)

</div>

## Install

```sh
go get github.com/inference-sim/blis-substrate@<version>
```

The packages are imported by path, such as
`github.com/inference-sim/blis-substrate/contracts/spec/scenario`. From the root of a
blis-substrate checkout, the validators check the repository's own data:

```sh
go run ./validate/cmd/validate-catalog catalog
go run ./validate/cmd/validate-registry registry
```

These pages are versioned. The selector in the header switches between release lines;
`dev` follows the main branch.
