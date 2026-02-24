---
name: Feature request
about: Propose a new capability for the simulator
title: ''
labels: 'enhancement'
assignees: ''

---

**What problem does this solve?**
Describe what's missing or limiting. What can't users or researchers do today?

**Proposed solution**
Describe the capability you'd like. Include a CLI example if applicable:
```bash
./simulation_worker run --model <model> --new-flag <value>
```

**Which components are affected?**
- [ ] Core simulator (`sim/`)
- [ ] Cluster simulation (`sim/cluster/`)
- [ ] Workload generation (`sim/workload/`)
- [ ] KV cache (`sim/kv/`)
- [ ] Decision tracing (`sim/trace/`)
- [ ] CLI (`cmd/`)
- [ ] New package needed

**Extension friction check**
- How many files would need to change to add this? (Estimate)
- Does this require a new interface, or can it extend an existing one?
- Does this affect any invariants (conservation, causality, determinism)?

**Alternatives considered**
What other approaches did you consider? Why is this one preferred?

**Relationship to existing work**
Does this relate to any open issues, the macro plan, or a design document?
