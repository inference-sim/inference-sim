# Archon Baseline Analysis

Run before designing a large feature. Understand the current architecture, blast radius, and health of the area you're about to change.

## Steps

1. **Health check** — understand the overall architecture:
```bash
archon-go health $REPO
```
Shows: cycles (should be none), god-modules (high fan-in + large surface), coupling table (fan-in, fan-out, surface, instability, blast radius per package).

2. **Impact analysis** — understand the blast radius of the package you'll change:
```bash
archon-go impact $REPO <target-package-path>
```
Shows: how many packages depend on this one (direct + transitive). Tells you what breaks if you change it.

3. **Evidence check** — are existing contracts test-covered?
```bash
archon-go evidence $REPO
```
Shows: which interfaces have contract tests, which don't. Tells you if you need to add coverage before changing things.

## What to use the output for

- **Health** informs where it's safe vs risky to add new packages. Low blast-radius areas are safer to extend.
- **Impact** tells you the scope of your change. High transitive dependents = more careful design needed.
- **Evidence** tells you if existing behavior is well-tested. If not, consider adding tests before changing things.

Feed these findings into your RFC — they inform the modeling decisions table (what to simplify, what to model carefully) and the trade-offs section (why this package placement, why these dependency directions).

## Example output

```
ARCHITECTURE HEALTH
  cycles: none — internal dependency graph is an acyclic DAG (healthy)
  god-modules (high fan-in + large surface): sim, latency, workload
  coupling (top by blast radius):
    package     fanIn  fanOut   surf   instab  blast
    sim             7       3    305     0.30      8  <god>
    kv              1       3     32     0.75      3
    cluster         1       5    315     0.83      2

BLAST RADIUS of .../sim/kv
  1 direct dependent(s), 3 total (transitive)
  direct:   cluster
  indirect: inference-sim, cmd
```

## Reference

For full archon CLI documentation: see the [archon repository](https://github.com/AI-native-Systems-Research/archon).
