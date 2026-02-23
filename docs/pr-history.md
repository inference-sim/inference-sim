# BLIS PR History and Design Documents

## Design Documents (per-feature)

### Active

- `docs/plans/2026-02-19-weighted-scoring-macro-plan.md`: **ACTIVE** — Composable scorer framework macro plan (2 PRs: PR 17-18)
- `docs/plans/2026-02-21-latency-model-extraction-design.md`: LatencyModel interface extraction design (#241)
- `docs/plans/latency-model-extraction-plan.md`: LatencyModel extraction implementation plan (#241)

### Archived (`docs/plans/archive/`)

- `2026-02-06-evolutionary-policy-optimization-design.md`: Full technical specification for cluster simulation extension (covers future PR11/PR15)
- `2026-02-19-weighted-scoring-evolution-design.md`: Design doc for composable scorer framework (specification species)
- `2026-02-13-simplification-assessment.md`: Architectural simplification assessment (constructor collapse, unified CLI, field privatization, interface dedup)
- `2026-02-16-workload-generator-design.md`: ServeGen-informed workload generator design (multi-client specs, arrival processes, calibration)
- `2026-02-18-hardening-antipattern-refactoring-design.md`: Hardening design — antipattern elimination, extension scenario analysis, modularity improvements
- `2026-02-20-seed-unification-design.md`: Seed unification design (#284)
- `pr12-architectural-predesign.md`: PR12 architectural pre-design — 6 binding design decisions for tiered KV cache (gold standard for decision records)
- `2026-02-13-mock-study-findings.md`: Post-PR3 mock study findings
- `2026-02-13-mock-study-implementation.md`: Mock study implementation plan

## Completed PRs

16 PRs across 6 phases extending BLIS to multi-replica cluster simulation (12 completed, 4 remaining):

- **PR1**: PartitionedRNG
- **PR2**: InstanceSimulator
- **PR3**: ClusterSimulator (shared-clock event loop, round-robin dispatch, metrics aggregation, golden dataset equivalence)
- **PR4**: Cluster control plane (online routing pipeline, SnapshotProvider, AdmissionPolicy with AlwaysAdmit + TokenBucket, cluster event queue)
- **PR5**: Architectural simplification (SimConfig struct, unified CLI path through ClusterSimulator, field privatization, AdmissionPolicy consolidated to `sim/admission.go`)
- **PR6**: RoutingPolicy interface (RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity templates; RoutingSnapshot bridge type)
- **PR7**: PriorityPolicy (ConstantPriority + SLOBasedPriority), InstanceScheduler (FCFS + PriorityFCFS + SJF), Priority field on Request
- **PR8**: RouterState bridge type, PolicyBundle YAML config, `--policy-config` CLI flag, **INTERFACE FREEZE**
- **PR9**: RawMetrics + FitnessResult, anomaly detection, pathological templates, `--fitness-weights`, **RESEARCH-READY CHECKPOINT**
- **PR10**: ServeGen-informed workload generator (`sim/workload/`), multi-client specs, arrival processes, calibration, real-mode HTTP client, `--workload-spec`
- **PR12**: TieredKVCache (GPU+CPU offload/reload), `KVStore` interface, `NewKVStore` factory, `--kv-cpu-blocks --kv-offload-threshold --kv-transfer-bandwidth`
- **PR13**: DecisionTrace, counterfactual analysis, TraceSummary, EvaluationResult, `--trace-level --counterfactual-k --summarize-trace`
- **PR17**: Composable scorer framework for weighted routing with stateless scorers
- **Hardening**: 6 phases (structural helpers, correctness fixes, metric fixes, invariant tests, input validation, modularity)
- **Bug fix**: Silent correctness bugs — `--total-kv-blocks` CLI override (#285), `--snapshot-refresh-interval` validation (#281), multi-client workload starvation (#278)

### Remaining

- **PR11**: AutoScaler
- **PR14**: P/D Disaggregation (depends on PR12)
- **PR15**: Framework Adapters
- **PR16**: Integration Tests
