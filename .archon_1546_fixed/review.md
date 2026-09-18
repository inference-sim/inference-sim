## ARCHON PR review — `70e9ba8` → `5e28e00b`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `ARCHITECTURAL_CHANGE`**

Architectural change — a package boundary moved; an architecture review is required.

1 edge− · 2 surface · 1 schema · 4 invariant · 1 contract

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
  sg8 -. "implements REMOVED" .-> sg2
  sg10 -->|"call, import"| sg2
  linkStyle 20 stroke:#cf222e,stroke-width:2px;
  classDef boundary fill:#eef3fb,stroke:#1a7f37,stroke-width:2px;
  classDef minor fill:#eef3fb,stroke:#0969da,stroke-width:1px,stroke-dasharray:4 3;
  classDef unchanged fill:#eef3fb,stroke:#57606a;
  class sg0 unchanged;
  class sg1 minor;
  class sg2 boundary;
  class sg3 minor;
  class sg4 unchanged;
  class sg5 unchanged;
  class sg6 unchanged;
  class sg7 unchanged;
  class sg8 boundary;
  class sg9 unchanged;
  class sg10 unchanged;
```

### Witness delta — full vs partial decoupling

**1 edge(s) fully decoupled; 1 edge(s) PARTIALLY decoupled (weakened)**

_Red dashed = connection fully removed · red solid = weakened (still coupled) · green = added/strengthened · blue = churned._

```mermaid
graph LR
  p0["cmd"]
  p1["sim"]
  p2["sim/saturation"]
  p3["sim/workload"]
  p2 -. "implements REMOVED" .-> p1
  p2 -->|"call WEAKENED"| p3
  p0 -->|"call CHURNED"| p2
  p0 -->|"import STRENGTHENED"| p2
  linkStyle 0 stroke:#cf222e,stroke-width:2px;
  linkStyle 1 stroke:#cf222e,stroke-width:2px;
  linkStyle 2 stroke:#0969da,stroke-width:2px;
  linkStyle 3 stroke:#1a7f37,stroke-width:2px;
```

| Edge | Kind | Status | Removed | Still coupled via |
|---|---|---|---|---|
| `sim/saturation → sim` | implements | **REMOVED** (full decoupling) | `Bank \|= BatchClassifier` | — |
| `sim/saturation → sim/workload` | call | **WEAKENED** (partial) | `NewBacklogClassifier` | `DefaultBacklogDriftConfig`, `NewBacklogDriftConfig` |
| `cmd → sim/saturation` | call | CHURNED | `Bank.Classify` | `AllDetectorNames`, `Bank.Close`, `BuildDetector`, `LoadSaturationConfig` _(+5 more symbol)_ |
| `cmd → sim/saturation` | import | STRENGTHENED | — | — |

### Interface-contract delta

_Green = implementer added · red dashed = implementer removed._

```mermaid
graph LR
  i0["sim.BatchClassifier"]
  m0("saturation.Bank")
  m0 -. implements .-> i0
  linkStyle 0 stroke:#cf222e,stroke-width:2px;
```

| Interface | + implementers | − implementers | uncovered (evidence gap) | contract test |
|---|---|---|---|---|
| `sim.BatchClassifier` | — | saturation.Bank | — | — |

### Invariants touched (guarded promises)

| Package | + added | ~ modified | − removed | promise on |
|---|---|---|---|---|
| `cmd` | TestResolveSaturation_FinalWindowErrors, TestResolveSaturation_FinalWindowResolutionOrder, TestSaturationStdout_FinalLabelShape, TestSaturationTracer_RunNoReportStillReturnsFinal | TestResolveSaturation_ConfigOrReportWithoutDetectors, TestSaturationTracer_AllEqualsExplicitList, TestSaturationTracer_BankWritesAllDetectors, TestSaturationTracer_DecoupledFromGlobals, TestSaturationTracer_SingleWritesTrace, TestSaturationTracer_SubsetMatchesRecordsUnderAll, TestSaturationTracer_ZeroRequestsWritesEmptyTrace, TestSaveResults_MetricsPrintedToStdout | TestSaturationBC8_StdoutByteIdenticalWithoutAndWithDetectors, TestSaturationTracer_TraceNoOpWhenNoReport | — |
| `sim` | — | TestBuildOutput_AdapterEventCountsSurface, TestBuildOutput_AdapterKeysSorted, TestBuildOutput_NoAdapters_OmitsBlock, TestBuildOutput_PerAdapterMetrics, TestColdLoadGate_INV8_NoDeadlockUnderCapacityPressure, TestColdLoadGate_LoadsSerializePerInstance, TestColdLoadGate_SameAdapterCoalesces, TestColdLoadGate_WarmIncursNoLoadLatency, TestNewSimulator_AdaptersWithoutCapacity, TestResidentAdapterSet_ActiveRun_Deterministic, TestResidentAdapterSet_CapacityBoundAcrossRun, TestResidentAdapterSet_InertWhenNoLoRA, TestResidentAdapterSet_MixedBaseAndAdapterTraffic, TestResidentAdapterSet_PreemptionDoesNotDoubleCountLoad, TestSaveResults_AlwaysEmitsHeader_ZeroCompletions, TestSaveResults_ConservationFields, TestSaveResults_DroppedUnservable_InJSON, TestSaveResults_IncludesIncompleteRequests, TestSaveResults_InstanceID_Default, TestSaveResults_InstanceID_Empty, TestSaveResults_InstanceID_InJSON, TestSaveResults_LengthCappedRequests_InJSON, TestSaveResults_NoWallClockFields, TestSaveResults_PerRequestITL_InMilliseconds, TestSaveResults_ZeroRuntime_NoInfinity, TestSimulator_Determinism_ByteIdenticalJSON | — | sim.ResidentAdapterSet |
| `cluster` | — | TestInstanceSimulator_EvictRequest_ReleasesAdapterPin | — | sim.Event |
| `saturation` | TestBank_RunActuallyReplaysEvents, TestBank_RunProducesNonEmptyTrace, TestReduceAll_EmptyInputEmptyMap, TestReduceAll_GroupsByDetector, TestReduceAll_PerGroupWindowing, TestReduceAll_SingleDetectorStillMap, TestReduceOne_Contracts, TestReduceOne_ExactWindowBoundaryInclusive, TestWriteCombinedReport_FinalBlock | TestBank_AllEqualsExplicitList, TestBank_Deterministic, TestBank_SubsetMatchesRecordsUnderAll, TestBank_ZeroRequestsEmptyTrace, TestE2E_ExtractorParity_ByteIdenticalTrace, TestE2E_ReplayComposite_WritesTrace, TestE2E_ReplayEmptyInput_WritesEmptyTrace, TestMetricsOutput_SaturationField, TestWriteCombinedReport_ByteIdentical, TestWriteCombinedReport_EmptyInput_WritesEmptyTrace | TestBank_ClassifyActuallyReplaysEvents, TestBank_SatisfiesBatchClassifierContract | sim.BatchClassifier, saturation.Detector, saturation.TraceSink |

### Schema (wire/DB data contract) changes

| Package | + fields | − fields |
|---|---|---|
| `saturation` | `CombinedReport.Final` | — |

### Public surface changes

| Package | + added | − removed |
|---|---|---|
| `sim` | — | `BatchClassifier` |
| `saturation` | `ReduceAll`, `ReduceOne`, `Bank.Run` | `NewBacklogDriftDetectorWithClassifier`, `Bank.Classify` |

