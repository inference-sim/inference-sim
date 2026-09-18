## ARCHON PR review — `18c2c9260cdf5af7007ffd39103c5304ecd805ac` → `pr-1524`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `ARCHITECTURAL_CHANGE`**

Architectural change — a package boundary moved; an architecture review is required.

2 edge+ · 1 surface · 2 invariant · 1 contract

### Component view

_Nodes: green = boundary moved · blue-dashed = surface/schema/invariant only · grey = unchanged. Edges: green = added · red-dashed = removed · grey = unchanged. ⟲ = dependency cycle._

```mermaid
graph TB
  subgraph sg0 ["(root)"]
    direction LR
    m0x0("root")
  end
  subgraph sg1 ["cmd"]
    direction LR
    m1x0("cmd")
  end
  subgraph sg2 ["sim"]
    direction LR
    m2x0("sim")
  end
  subgraph sg3 ["sim/cluster"]
    direction LR
    m3x0("cluster")
  end
  subgraph sg4 ["sim/internal"]
    direction LR
    m4x0("hash")
    m4x1("testutil")
    m4x2("tokenid")
    m4x3("util")
  end
  subgraph sg5 ["sim/kv"]
    direction LR
    m5x0("kv")
  end
  subgraph sg6 ["sim/latency"]
    direction LR
    m6x0("latency")
  end
  subgraph sg7 ["sim/lora"]
    direction LR
    m7x0("lora")
  end
  subgraph sg8 ["sim/saturation"]
    direction LR
    m8x0("saturation")
  end
  subgraph sg9 ["sim/trace"]
    direction LR
    m9x0("trace")
  end
  subgraph sg10 ["sim/workload"]
    direction LR
    m10x0("workload")
  end

  sg0 -->|"call, import"| sg1
  sg1 -->|"call, import"| sg2
  sg1 -->|"call, import"| sg3
  sg1 -->|"call, import"| sg6
  sg1 -->|"import"| sg7
  sg1 -->|"call, import"| sg8
  sg1 -->|"call, import"| sg9
  sg1 -->|"call, import"| sg10
  sg2 -->|"call, import"| sg4
  sg3 -. "call, implements, import" .-> sg2
  sg3 -->|"call, import"| sg5
  sg3 -->|"call, import"| sg6
  sg3 -->|"call, import"| sg9
  sg3 -->|"import"| sg10
  sg5 -. "call, implements, import" .-> sg2
  sg5 -->|"call, import"| sg4
  sg6 -. "call, implements, import" .-> sg2
  sg7 -. "implements, import" .-> sg2
  sg8 -->|"import"| sg2
  sg8 -->|"call, import"| sg10
  sg8 -. "implements ADDED" .-> sg2
  sg10 -->|"call, import"| sg2
  linkStyle 20 stroke:#1a7f37,stroke-width:2px;
  classDef boundary fill:#eef3fb,stroke:#1a7f37,stroke-width:2px;
  classDef minor fill:#eef3fb,stroke:#0969da,stroke-width:1px,stroke-dasharray:4 3;
  classDef unchanged fill:#eef3fb,stroke:#57606a;
  class sg0 unchanged;
  class sg1 minor;
  class sg2 boundary;
  class sg3 unchanged;
  class sg4 unchanged;
  class sg5 unchanged;
  class sg6 unchanged;
  class sg7 unchanged;
  class sg8 boundary;
  class sg9 unchanged;
  class sg10 unchanged;
```

### Witness delta — full vs partial decoupling

_Red dashed = connection fully removed · red solid = weakened (still coupled) · green = added/strengthened · blue = churned._

```mermaid
graph LR
  p0["cmd"]
  p1["sim"]
  p2["sim/saturation"]
  p0 -->|"call STRENGTHENED"| p2
  p2 -->|"import STRENGTHENED"| p1
  p2 -->|"implements ADDED"| p1
  linkStyle 0 stroke:#1a7f37,stroke-width:2px;
  linkStyle 1 stroke:#1a7f37,stroke-width:2px;
  linkStyle 2 stroke:#1a7f37,stroke-width:2px;
```

| Edge | Kind | Status | Removed | Still coupled via |
|---|---|---|---|---|
| `cmd → sim/saturation` | call | STRENGTHENED | — | — |
| `sim/saturation → sim` | import | STRENGTHENED | — | — |
| `sim/saturation → sim` | implements | ADDED | — | — |

### Interface-contract delta

_Green = implementer added · red dashed = implementer removed._

```mermaid
graph LR
  i0["sim.BatchClassifier"]
  m0("saturation.Bank")
  m0 -->|implements| i0
  linkStyle 0 stroke:#1a7f37,stroke-width:2px;
```

| Interface | + implementers | − implementers | uncovered (evidence gap) | contract test |
|---|---|---|---|---|
| `sim.BatchClassifier` | saturation.Bank | — | — | `TestBank_SatisfiesBatchClassifierContract` |

### Invariants touched (guarded promises)

| Package | + added | ~ modified | − removed | promise on |
|---|---|---|---|---|
| `cmd` | TestResolveSaturation_AllMixedWithNames, TestResolveSaturation_AllUsesBank, TestResolveSaturation_BankSelectedBlockAccepted, TestResolveSaturation_BankTrailingCommaEnforcesOwnership, TestResolveSaturation_BankUnselectedBlockErrors, TestResolveSaturation_EmptySelection, TestResolveSaturation_SingleDetectorBlockOwnership, TestResolveSaturation_SubsetListUsesBank, TestResolveSaturation_UnknownNameInList, TestSaturationBC8_StdoutByteIdenticalWithoutAndWithDetectors, TestSaturationTracer_AllEqualsExplicitList, TestSaturationTracer_BankWritesAllDetectors, TestSaturationTracer_DecoupledFromGlobals, TestSaturationTracer_SingleWritesTrace, TestSaturationTracer_SubsetMatchesRecordsUnderAll, TestSaturationTracer_TraceNoOpWhenNoReport, TestSaturationTracer_ZeroRequestsWritesEmptyTrace | TestResolveSaturation_ConfigOrReportWithoutDetectors, TestResolveSaturation_Off, TestResolveSaturation_UnknownName, TestResolveSaturation_UnwritableReportPath, TestResolveSaturation_ValidSingleDetector | TestResolveSaturation_BankRejected, TestResolveSaturation_CompositeWithEmptyConfig, TestRunSaturationTrace_NoOpWhenOff, TestRunSaturationTrace_WritesTrace | saturation.Detector |
| `saturation` | TestBank_AllEqualsExplicitList, TestBank_ClassifyActuallyReplaysEvents, TestBank_Deterministic, TestBank_SatisfiesBatchClassifierContract, TestBank_SubsetMatchesRecordsUnderAll, TestBank_ZeroRequestsEmptyTrace, TestNewBank_AllAcceptsEveryBlock, TestNewBank_CanonicalOrderAndDedup, TestNewBank_EmptySelectionErrors, TestNewBank_SelectedBlockAccepted, TestNewBank_SelectedBlockValueErrorSurfaces, TestNewBank_UnknownNameErrors, TestNewBank_UnselectedBlockErrors | — | — | — |

### Public surface changes

| Package | + added | − removed |
|---|---|---|
| `saturation` | `AllDetectorNames`, `NewBank`, `Bank.Classify`, `Bank.Close`, `Bank` | — |

