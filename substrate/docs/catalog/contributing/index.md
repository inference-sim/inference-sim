# Contributing

A change to the catalog is a change to data. Each change needs the new figures and the source each one came from, recorded where a reviewer can check it.

One rule decides what can be added: **an entry holds only declared facts**, figures a person can read off a published source such as a model's `config.json` or a vendor datasheet. A number a measurement would revise, such as an achieved bandwidth or a fitted efficiency, is a coefficient and belongs in the [registry](../../registry/index.md). A setting chosen for one run, such as a GPU count or a parallelism degree, belongs in that run's scenario. [Declared, learned, chosen](../concepts/declared-facts.md) explains the rule.

## Recipes

Paths on these pages are relative to `catalog/` in blis-substrate, and commands run from the repository root.

| To add | Read |
| :--- | :--- |
| a model | [Add a model](add-model.md) |
| a chip, a fabric or a storage tier | [Add a chip, fabric or storage tier](add-hardware.md) |
| a workload | [Add a workload](add-workload.md) |

**To correct an entry**, edit the figure and the comment that cites its source, and say in the pull request what was wrong. To move a model to a newer upstream revision, repeat steps 2 to 4 of [Add a model](add-model.md); the graph's recorded digest changes with the config.

## The workflow

1. Branch from `main` and make the change.
2. Run the [checks](checks.md) locally.
3. Open a pull request. In the description, cite the source of every figure you added or changed. CI runs the same checks.
4. If the change adds an entry, preview the site. A new model gets its own reference page, which is a convenient way to review its derived graph. See [Preview locally](../../contributing/documentation.md#preview-locally).

The catalog is released with the rest of blis-substrate; [Documentation and releases](../../contributing/documentation.md) describes how a release is cut and its site published.

## Changes outside the catalog directory

- **A new field, or a new rule about an existing field**, belongs in `contracts/spec`, which defines the format of every catalog file. It ships with its validator, a failing example for each way it can be wrong, and regenerated reference documentation (SUB-2), in the same pull request as the catalog entries that use it.
- **A new way of pricing something** belongs in `kernel/`, the cost model, in the same pull request if the entry needs it.
- **A change that also touches another BLIS repository**, such as an engine or inference-sim, follows [Changes across repositories](../../contributing/changes.md).

A new model architecture built from operations the graph format already defines needs none of these. It needs a handler in the deriver, `catalog/scripts/derive_graph.py`; see [Add a model](add-model.md#if-the-architecture-is-new).
