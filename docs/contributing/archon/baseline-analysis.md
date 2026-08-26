# Archon Baseline Analysis

Run before designing a large feature to understand the current architecture, blast radius, and health of the area you'll change.

**Requires:** archon v0.2.0+. See [archon README](https://github.com/AI-native-Systems-Research/archon) for build instructions, full command reference, and examples.

## What to run

```bash
archon-go health $REPO                          # cycles, god-modules, blast radius
archon-go impact $REPO <target-package-path>    # what depends on this package?
archon-go evidence $REPO                        # are contracts test-covered?
```

## What to use the output for

- **Health** → informs where it's safe vs risky to add packages
- **Impact** → tells you scope of your change (high transitive deps = careful design)
- **Evidence** → tells you if existing behavior is well-tested before you change it

Feed these findings into your RFC (modeling decisions, trade-offs).
