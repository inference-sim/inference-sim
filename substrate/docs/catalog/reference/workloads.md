# Workloads

[[catalog.count.workloads]] workloads, in two forms. A **shape** says how long requests are, not how many arrive; the arrival rate or concurrency is set by each run. A **replay workload** names a load generator's trace-replay scenario over a pinned corpus; the corpus decides the requests, and each run sets the number of lanes.

## Shapes

<figure markdown="1">
[[catalog.workloads.figure]]
<figcaption>Tokens per request, on a logarithmic scale. The thin line spans the stated minimum to maximum, the thick bar one standard deviation either side of the mean (clipped to those bounds), and the dot marks the mean.</figcaption>
</figure>

[[catalog.workloads.table]]

A file states a mean, a standard deviation and bounds, not the form of the distribution. That is the sampler's choice: inference-sim draws from a normal distribution with the stated mean and standard deviation, clamped to the bounds.

## Replay workloads

Replay workloads in this release: [[catalog.count.replays]]. A file whose top-level key is `replay:` states the scenario, the harness release it reproduces, the corpus by repository, revision and digest, and the scenario's settings. It does not state a lane count (`concurrency`): a scenario that binds the workload states it, with `replay: {workload: <name>, concurrency: C}`.

[[catalog.workloads.replays]]

The corpus is not in the catalog. A consumer fetches the file at the stated revision and checks its SHA-256 before replaying it.

[[catalog.workloads.replays_corpus]]

### Sources

Each replay file records its provenance in its `_comment` keys. They are reproduced here as written.

[[catalog.workloads.replays_notes]]
