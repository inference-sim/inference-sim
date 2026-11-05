---
hide:
  - navigation
---

# BLIS Catalog

<p class="lede">The catalog holds the facts that <a href="https://github.com/inference-sim/inference-sim">BLIS</a> simulates with: what a model is, what a chip can do, what a network fabric can carry, what traffic looks like, and what a storage tier costs. Its rule is that every entry is a <em>declared fact</em>, written down from a published source such as a model's own configuration file or a vendor datasheet, and not fitted to measurements or chosen for a particular run. <a href="concepts/declared-facts/#where-the-catalog-falls-short-of-its-rule">A few entries</a> fall short of the rule today.</p>

This site documents [[catalog.release]].

## What is in it

| Directory | Holds | Entries |
| :--- | :--- | ---: |
| [`models/`](reference/models/index.md) | Models: the vendor's `config.json`, where it came from, and the model graph derived from it | [[catalog.count.models]] |
| [`hardware/`](reference/hardware.md) | Chips: peak rates, memory, intra-node bandwidth | [[catalog.count.chips]] |
| [`networks/`](reference/networks.md) | Fabrics: what an inter-node network can carry | [[catalog.count.fabrics]] |
| [`workloads/`](reference/workloads.md) | Workloads: distributions of prompt and output length, and trace-replay scenarios | [[catalog.count.workloads]] |
| [`devices/`](reference/storage.md) | Storage tiers: where a KV cache can be offloaded, and at what cost | [[catalog.count.tiers]] |

The catalog is plain YAML and JSON. Reading it needs no tools.

## Where to start

| To | Read |
| :--- | :--- |
| run BLIS against the catalog | [Using](using/index.md) |
| add or correct an entry | [Contributing](contributing/index.md) |
| understand why the catalog is shaped as it is | [Concepts](concepts/index.md) |
| look up a figure | [Reference](reference/index.md) |

The reference tables and figures are generated from the files in this release when the site is built.
