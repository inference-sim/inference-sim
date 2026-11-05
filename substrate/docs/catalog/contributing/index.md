# Contributing

A change to the catalog is a change to data. Each change needs the new figures and the source each one came from, recorded where a reviewer can check it.

One rule decides what can be added: **an entry holds only declared facts**, figures a person can read off a published source such as a model's `config.json` or a vendor datasheet. A number a measurement would revise, such as an achieved bandwidth or a fitted efficiency, is a coefficient and belongs in [blis-registry](https://github.com/inference-sim/blis-registry). A setting chosen for one run, such as a GPU count or a parallelism degree, belongs in that run's scenario. [Declared, learned, chosen](../concepts/declared-facts.md) explains the rule.

## Recipes

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
4. If the change adds an entry, preview the site. A new model gets its own reference page, which is a convenient way to review its derived graph. See [Preview locally](releasing.md#preview-locally).

Maintainers tag releases on `main`; [Releasing and the docs site](releasing.md) describes how.

## Changes that start elsewhere

- **A new field, or a new rule about an existing field**, belongs in [blis-schemas](https://github.com/inference-sim/blis-schemas), which defines the format of every catalog file. The catalog adopts it by raising the validator version its CI pins.
- **A new way of pricing something** belongs in [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel).
- **A change that spans several repositories** follows [Changes across the repositories](cross-repo-changes.md), which links the canonical process and lists the catalog's own steps.

A new model architecture built from operations the schemas already define needs neither. It needs a handler in this repository's deriver; see [Add a model](add-model.md#if-the-architecture-is-new).
