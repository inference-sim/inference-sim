## ARCHON PR review — `d77764f520568b6c67616ca178b214fe288be7fd` → `pr-1567`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `REVIEW_ARCHITECTURE`**

REVIEW_ARCHITECTURE — a package boundary moved; an architecture pass is required.

1 pkg− · 4 edge− · 3 surface · 1 schema · 2 invariant

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
  n10 --> n2
  classDef boundary fill:#eef3fb,stroke:#1a7f37,stroke-width:2px;
  classDef minor fill:#eef3fb,stroke:#0969da,stroke-width:1px,stroke-dasharray:4 3;
  classDef unchanged fill:#eef3fb,stroke:#57606a;
  class n0 unchanged;
  class n1 unchanged;
  class n2 minor;
  class n3 unchanged;
  class n4 unchanged;
  class n5 unchanged;
  class n6 unchanged;
  class n7 unchanged;
  class n8 boundary;
  class n9 unchanged;
  class n10 boundary;
```

### Witness delta — full vs partial decoupling

**2 edge(s) fully decoupled; 0 edge(s) PARTIALLY decoupled (weakened)**

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

---

Artifacts: [`component.dot`](component.dot) · [`component.mmd`](component.mmd) · [`contract.md`](contract.md) · [`witness.dot`](witness.dot)
