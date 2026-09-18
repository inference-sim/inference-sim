## ARCHON PR review — `d77764f520568b6c67616ca178b214fe288be7fd` → `pr-1567`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `ARCHITECTURAL_CHANGE`**

Architectural change — a package boundary moved; an architecture review is required.

1 pkg− · 4 edge− · 3 surface · 1 schema · 2 invariant

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
  sg8 -. "call, import REMOVED" .-> sg10
  sg10 -->|"call, import"| sg2
  linkStyle 19 stroke:#cf222e,stroke-width:2px;
  classDef boundary fill:#eef3fb,stroke:#1a7f37,stroke-width:2px;
  classDef minor fill:#eef3fb,stroke:#0969da,stroke-width:1px,stroke-dasharray:4 3;
  classDef unchanged fill:#eef3fb,stroke:#57606a;
  class sg0 unchanged;
  class sg1 unchanged;
  class sg2 minor;
  class sg3 unchanged;
  class sg4 unchanged;
  class sg5 unchanged;
  class sg6 unchanged;
  class sg7 unchanged;
  class sg8 boundary;
  class sg9 unchanged;
  class sg10 boundary;
```

### Witness delta — full vs partial decoupling

**2 edge(s) fully decoupled; 0 edge(s) PARTIALLY decoupled (weakened)**

_Red dashed = connection fully removed · red solid = weakened (still coupled) · green = added/strengthened · blue = churned._

```mermaid
graph LR
  p0["sim/saturation"]
  p1["sim/workload"]
  p0 -. "call REMOVED" .-> p1
  p0 -. "import REMOVED" .-> p1
  linkStyle 0 stroke:#cf222e,stroke-width:2px;
  linkStyle 1 stroke:#cf222e,stroke-width:2px;
```

| Edge | Kind | Status | Removed | Still coupled via |
|---|---|---|---|---|
| `sim/saturation → sim/workload` | call | **REMOVED** (full decoupling) | `DefaultBacklogDriftConfig`, `NewBacklogDriftConfig` | — |
| `sim/saturation → sim/workload` | import | **REMOVED** (full decoupling) | `backlog_drift.go`, `config.go` | — |

### Interface-contract delta

_No interface-contract membership changed._

### Invariants touched (guarded promises)

| Package | + added | ~ modified | − removed | promise on |
|---|---|---|---|---|
| `saturation` | TestBacklogDriftConfig_Validation_CIOutOfRange, TestBacklogDriftConfig_Validation_NaNPeakRatio, TestBacklogDriftConfig_Validation_NegativeMinWindows, TestBacklogDriftConfig_Validation_ValidConfig, TestBacklogDriftConfig_Validation_ZeroWindow | — | — | — |
| `workload` | — | — | TestAnalyzeBacklogDriftWithClassifier_NilClassifier_DefaultsToSlopeBased, TestAnalyzeBacklogDriftWithClassifier_OrchestratorFallback_AllInjectWindows, TestAnalyzeBacklogDrift_AbsoluteTimestamps, TestAnalyzeBacklogDrift_AbsoluteTimestamps_ClassificationPreserved, TestAnalyzeBacklogDrift_AllExcluded, TestAnalyzeBacklogDrift_EndToEnd_PERSISTENTLY_SATURATED, TestAnalyzeBacklogDrift_EndToEnd_UNSATURATED, TestAnalyzeBacklogDrift_InsufficientData, TestBacklogDriftConfig_NewBacklogDriftConfig_DrainRatioParamValidation, TestBacklogDriftConfig_NewBacklogDriftConfig_ValidatesDrainRatioRange, TestBacklogDriftConfig_Validation_CIOutOfRange, TestBacklogDriftConfig_Validation_NaNPeakRatio, TestBacklogDriftConfig_Validation_NegativeMinWindows, TestBacklogDriftConfig_Validation_ValidConfig, TestBacklogDriftConfig_Validation_ZeroWindow, TestClassifyBacklogDrift_PERSISTENTLY_SATURATED, TestClassifyBacklogDrift_TRANSIENT_BACKLOG, TestClassifyBacklogDrift_UNSATURATED, TestComputeWindowMetrics_ActiveCount_AtBoundaries, TestComputeWindowMetrics_Identity_DeltaBacklogEqualsEnterMinusLeft, TestComputeWindowMetrics_UnreasonablyLargeDuration, TestDrainRatioClassifier_AllWarmup_Unsaturated, TestDrainRatioClassifier_InteriorEmptyWindow_RemainsInject, TestDrainRatioClassifier_LastArrivalWindow_TrailingDrainExcluded, TestDrainRatioClassifier_NaNSkipCount_SurfacesInNote, TestDrainRatioClassifier_NoArrivals_Unsaturated, TestDrainRatioClassifier_PersistentlySaturated, TestDrainRatioClassifier_TransientBacklog, TestDrainRatioClassifier_Unsaturated, TestFitSlopeRegression_FlatLine, TestFitSlopeRegression_NegativeSlope, TestFitSlopeRegression_PositiveSlope, TestIsValidBacklogClassifier_RegistryContents, TestNewBacklogClassifier_DefaultIsDrainRatio, TestNewBacklogClassifier_FactoryRegistryAgreement, TestNewBacklogClassifier_PanicsOnUnknown, TestNewBacklogClassifier_SlopeBased, TestProperty_P1_Determinism, TestProperty_P2_MonotonicityExtremes, TestProperty_P3_DrainPhaseInvariance, TestProperty_P4_WarmupRobustness, TestProperty_P5_Conservation, TestProperty_P6_CrossClassifierAgreement, TestReadBacklogDriftReportJSON_InvalidFile, TestRequestsToIntervals_Conservation_QueuedAndRunningBothCounted, TestRequestsToIntervals_Eligibility_ThreeCases, TestRequestsToIntervals_EmptyInput_ReturnsEmpty, TestRequestsToIntervals_StateQueued_IncludedAsHorizonTruncated, TestSaturationClassification_ManualScenarios, TestSaturationProgression_Demonstration, TestSaturationProgression_RealWorkloads, TestSaturationProgression_TransitionBoundaries, TestWriteBacklogDriftReportJSON_RoundTrip, TestWriteBacklogDriftReportJSON_SanitizesNaN | workload.BacklogClassifier |

### Schema (wire/DB data contract) changes

| Package | + fields | − fields |
|---|---|---|
| `workload` | — | `BacklogDriftReport.Classification`, `BacklogDriftReport.FinalBacklog`, `BacklogDriftReport.InitialBacklog`, `BacklogDriftReport.MeanInFlight`, `BacklogDriftReport.Note`, `BacklogDriftReport.PeakInFlight`, `BacklogDriftReport.Recommendation`, `BacklogDriftReport.Slope`, `BacklogDriftReport.SlopeLower`, `BacklogDriftReport.SlopeUpper`, `BacklogDriftReport.Windows` |

### Public surface changes

| Package | + added | − removed |
|---|---|---|
| `sim` | — | `IsValidBacklogClassifier`, `ValidBacklogClassifierNames` |
| `saturation` | `DefaultBacklogDriftConfig`, `NewBacklogDriftConfig`, `BacklogDriftConfig` | — |
| `workload` | — | `AnalyzeBacklogDrift`, `AnalyzeBacklogDriftWithClassifier`, `DefaultBacklogDriftConfig`, `NewBacklogClassifier`, `NewBacklogDriftConfig`, `ReadBacklogDriftReportJSON`, `RequestsToIntervals`, `WriteBacklogDriftReportJSON`, `BacklogClassifier`, `BacklogDriftConfig`, `BacklogDriftReport`, `RequestInterval`, `SlopeStats`, `WindowMetrics` |

