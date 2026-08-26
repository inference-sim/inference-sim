# Archon Baseline Analysis

Run before designing a large feature to understand the current architecture, blast radius, and health of the area you'll change.

**Requires:** archon v0.2.0+. Build and usage: see [archon README](https://github.com/AI-native-Systems-Research/archon#quick-start).

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

## Full reference

- Command details + example output: [archon README — Flow 1](https://github.com/AI-native-Systems-Research/archon#flow-1-pr-review-existing-code-no-setup)
- Real end-to-end walkthrough: [demo/flow3-blis-design](https://github.com/AI-native-Systems-Research/archon/blob/main/demo/flow3-blis-design/README.md)
