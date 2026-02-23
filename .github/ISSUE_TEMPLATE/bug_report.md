---
name: Bug report
about: Report a correctness bug, silent failure, or wrong simulation result
title: 'Bug: '
labels: 'bug'
assignees: ''

---

**Describe the bug**
What is wrong? Be specific about the incorrect behavior.

**CLI command to reproduce**
```bash
./simulation_worker run --model <model> [flags]
```

**Expected behavior**
What should happen? Reference an invariant if applicable:
- Request conservation: injected == completed + queued + running
- Request lifecycle: queued -> running -> completed (no invalid transitions)
- KV block conservation: allocated + free == total
- Causality: arrival <= schedule <= completion
- Clock monotonicity: clock never decreases
- Determinism: same seed produces identical output
- Signal freshness: routing snapshot signals have tiered freshness
- Work-conserving: simulator must not idle while work is waiting

**Actual behavior**
What happens instead? Include relevant log output or JSON metrics.

**Which invariant is violated?**
- [ ] Request conservation
- [ ] Request lifecycle
- [ ] KV block conservation
- [ ] Causality
- [ ] Clock monotonicity
- [ ] Determinism
- [ ] Signal freshness
- [ ] Work-conserving
- [ ] None / unknown

**Environment**
- OS: [e.g., macOS 15, Ubuntu 22.04]
- Go version: [e.g., 1.23]
- Commit hash: [e.g., daf24cf]

**Additional context**
How was this discovered? (Code review, test failure, user report, audit)
