# Using the catalog

BLIS reads the catalog from a local directory you name. It never downloads catalog data and never writes to it, so a simulation's catalog data comes only from that directory.

!!! note "Which BLIS these pages describe"
    These pages describe inference-sim's `blis-latency-kernel` backend, introduced in [inference-sim#1851](https://github.com/inference-sim/inference-sim/pull/1851). On that backend the kernel reads models, chips, fabrics and storage tiers through [blis-schemas](https://github.com/inference-sim/blis-schemas), in the formats the [File formats](../reference/formats.md) page describes. inference-sim also reads some catalog files itself, with readers of its own; [Which program reads which file](point-blis.md#which-program-reads-which-file) lists them, and why the release you clone must be the one inference-sim declares compatible.

To use the catalog:

1. **Clone it at a release tag** and point BLIS at the clone. [Point BLIS at the catalog](point-blis.md) shows the commands, how a run names the entries it needs, and how to record which catalog a result came from.
2. **Pin the release your tools are tested against.** Consumers parse catalog files strictly, so a newer release can stop an older tool at startup. [Releases and pinning](releases.md) explains why, and how to upgrade.
3. **Experiment in a scratch clone, not in place.** To try a hypothetical model or chip, edit a copy. [Experiment with a scratch clone](scratch-clone.md) shows how.
