## ARCHON PR review — `70e9ba855ecdf00e08c40984a17c1934b5aa8d77` → `pr-1546`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `ARCHITECTURAL_CHANGE`**

Architectural change — a package boundary moved; an architecture review is required.

1 edge− · 2 surface · 1 schema · 4 invariant · 1 contract

### Component view

_Green = boundary moved · blue-dashed = surface/schema/invariant only · grey = unchanged. ⟲ marks a dependency cycle._

```mermaid
graph LR
  n0["(root)"]
  n1["cmd"]
  n2["sim"]
  n3["sim/cluster"]
  n4["sim/internal"]
  n5["sim/kv"]
  n6["sim/latency"]
  n7["sim/lora"]
  n8["sim/saturation"]
  n9["sim/trace"]
  n10["sim/workload"]
  n0 --> n1
  n1 --> n2
  n1 --> n3
  n1 --> n6
  n1 --> n7
  n1 --> n8
  n1 --> n9
  n1 --> n10
  n2 --> n4
  n3 -. implements .-> n2
  n3 --> n2
  n3 --> n5
  n3 --> n6
  n3 --> n9
  n3 --> n10
  n5 -. implements .-> n2
  n5 --> n2
  n5 --> n4
  n6 -. implements .-> n2
  n6 --> n2
  n7 -. implements .-> n2
  n7 --> n2
  n8 --> n2
  n8 --> n10
  n10 --> n2
  classDef boundary fill:#eef3fb,stroke:#1a7f37,stroke-width:2px;
  classDef minor fill:#eef3fb,stroke:#0969da,stroke-width:1px,stroke-dasharray:4 3;
  classDef unchanged fill:#eef3fb,stroke:#57606a;
  class n0 unchanged;
  class n1 minor;
  class n2 boundary;
  class n3 minor;
  class n4 unchanged;
  class n5 unchanged;
  class n6 unchanged;
  class n7 unchanged;
  class n8 boundary;
  class n9 unchanged;
  class n10 unchanged;
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
| `saturation` | TestBank_RunActuallyReplaysEvents, TestBank_RunProducesNonEmptyTrace, TestReduceAll_EmptyInputEmptyMap, TestReduceAll_GroupsByDetector, TestReduceAll_PerGroupWindowing, TestReduceAll_SingleDetectorStillMap, TestReduceOne_AllLevelsOutOfRange_DefaultsToStable, TestReduceOne_Contracts, TestReduceOne_ExactWindowBoundaryInclusive, TestWriteCombinedReport_FinalBlock | TestBank_AllEqualsExplicitList, TestBank_Deterministic, TestBank_SubsetMatchesRecordsUnderAll, TestBank_ZeroRequestsEmptyTrace, TestE2E_ExtractorParity_ByteIdenticalTrace, TestE2E_ReplayComposite_WritesTrace, TestE2E_ReplayEmptyInput_WritesEmptyTrace, TestMetricsOutput_SaturationField, TestWriteCombinedReport_ByteIdentical, TestWriteCombinedReport_EmptyInput_WritesEmptyTrace | TestBank_ClassifyActuallyReplaysEvents, TestBank_SatisfiesBatchClassifierContract | sim.BatchClassifier, saturation.Detector, saturation.TraceSink |

### Schema (wire/DB data contract) changes

| Package | + fields | − fields |
|---|---|---|
| `saturation` | `CombinedReport.Final` | — |

### Public surface changes

| Package | + added | − removed |
|---|---|---|
| `sim` | — | `BatchClassifier` |
| `saturation` | `ReduceAll`, `ReduceOne`, `Bank.Run` | `NewBacklogDriftDetectorWithClassifier`, `Bank.Classify` |

