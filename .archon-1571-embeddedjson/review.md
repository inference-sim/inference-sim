## ARCHON PR review — `d77764f520568b6c67616ca178b214fe288be7fd` → `pr-1571`

_Deterministic · no LLM · package-altitude architecture._

**Verdict: `NO_CHANGE`**

✓ No architectural change. Internal-only — fast-track eligible.

_Note: 4 guarded promise(s) (invariant / schema) also changed within the existing boundary — see `review.json`._

<details>
<summary><code>review.json</code></summary>

```json
{
  "schema": "archon.pr-review/v1",
  "repo": "/Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/codeboarding/main-repo-blis",
  "base": "d77764f520568b6c67616ca178b214fe288be7fd",
  "head": "pr-1571",
  "labelA": "d77764f520568b6c67616ca178b214fe288be7fd",
  "labelB": "pr-1571",
  "verdict": "NO_CHANGE",
  "summary": "✓ No architectural change. Internal-only — fast-track eligible.",
  "emptyAtPackageAltitude": true,
  "counts": {
    "packagesAdded": 0,
    "packagesRemoved": 0,
    "edgesAdded": 0,
    "edgesRemoved": 0,
    "surfaceChanged": 0,
    "schemaChanged": 1,
    "invariants": 3,
    "contracts": 0,
    "violations": 0,
    "witnessesFullyDecoupled": 0,
    "witnessesPartiallyDecoupled": 0
  },
  "invariants": [
    {
      "package": "github.com/inference-sim/inference-sim/cmd",
      "added": [
        "TestRunCmd_DeprecatedBatchFlagAliases_Registered",
        "TestRunCmd_MaxNumBatchedTokens_FlagRegistered",
        "TestRunCmd_MaxNumSeqs_FlagRegistered"
      ],
      "removed": [
        "TestRunCmd_MaxRunningReqs_FlagRegistered",
        "TestRunCmd_MaxScheduledTokens_FlagRegistered"
      ],
      "modified": [
        "TestBothCommands_SimConfigFlagsHaveIdenticalDefaults",
        "TestINV13_RunReplayParity_PD_CLI",
        "TestReplayCmd_AnomalyBlock_TimedOutRequests",
        "TestReplayCmd_AutoscalerBundleFatal",
        "TestReplayCmd_AutoscalerFlagFatal",
        "TestReplayCmd_EndToEnd_TrainedPhysicsMode",
        "TestReplayCmd_NodePoolsBundleFatal",
        "TestReplayCmd_PDTopologyFatal",
        "TestReplayCmd_PD_BasicSmoke",
        "TestReplayCmd_SimConfigFlags_Registered",
        "TestReplayCmd_TraceOutput_Determinism",
        "TestReplayCmd_TraceOutput_FilesCreated",
        "TestReplayCmd_TraceOutput_NoOp",
        "TestRunCmd_MetricsPath_WritesMetricsOutput",
        "TestRunCmd_TraceOutput_RecordCountMatchesRequests"
      ]
    },
    {
      "package": "github.com/inference-sim/inference-sim/sim",
      "modified": [
        "TestNewBatchConfig_FieldEquivalence",
        "TestNewBatchConfig_PanicsOnInvalid",
        "TestNewSimulator_BatchConfigValidation",
        "TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption",
        "TestPreempt_EmptyBatch_ReturnsFalse",
        "TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse",
        "TestPreemption_FCFS_EvictsTail",
        "TestPreemption_Priority_EmptyBatch_NoPanic",
        "TestPreemption_Priority_EvictsLeastUrgent",
        "TestPreemption_Priority_KVConservation",
        "TestPreemption_Priority_MultiEvictionOrdering",
        "TestPreemption_Priority_Phase1Completeness",
        "TestPreemption_Priority_SelfPreemption",
        "TestPreemption_Priority_TiebreakByLatestArrival",
        "TestRegenGoldenDataset",
        "TestSimulator_GoldenDataset",
        "TestVLLMBatchFormation_BatchSizeEnforced",
        "TestVLLMBatchFormation_CircuitBreaker",
        "TestVLLMBatchFormation_ImplementsInterface",
        "TestVLLMBatchFormation_KVAllocationFailure_StopsDequeue",
        "TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Decode",
        "TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Phase2",
        "TestVLLMBatchFormation_MaxModelLen_Zero_NoClamp",
        "TestVLLMBatchFormation_Phase1_EvictedNotRevisited",
        "TestVLLMBatchFormation_PreemptionReleasesKV",
        "TestVLLMBatchFormation_PreemptionStopsDequeue",
        "TestVLLMBatchFormation_TokenBudgetEnforced",
        "TestVLLMBatchFormation_ZeroInputRequest_SkipsDecodeOnlyPath"
      ],
      "guardedContracts": [
        "github.com/inference-sim/inference-sim/sim.BatchFormation",
        "github.com/inference-sim/inference-sim/sim.KVStore",
        "github.com/inference-sim/inference-sim/sim.LatencyModel"
      ]
    },
    {
      "package": "github.com/inference-sim/inference-sim/sim/cluster",
      "modified": [
        "TestAddPending_StoresSimCfg",
        "TestClusterSimulator_OverloadConservation",
        "TestClusterSimulator_SchedulerLiveness",
        "TestClusterSimulator_SingleInstance_GoldenEquivalence",
        "TestClusterSimulator_SingleInstance_GoldenInvariants",
        "TestDisaggregation_TTFT_IncludesDecodeQueueWait",
        "TestInstanceSimulator_GoldenDataset_Equivalence",
        "TestInstanceSimulator_GoldenDataset_Invariants"
      ]
    }
  ],
  "schemaChanges": [
    {
      "package": "github.com/inference-sim/inference-sim/sim/internal/testutil",
      "added": [
        {
          "kind": "field",
          "name": "GoldenTestCase.MaxNumBatchedTokens",
          "sig": "int64"
        },
        {
          "kind": "field",
          "name": "GoldenTestCase.MaxNumSeqs",
          "sig": "int64"
        }
      ],
      "removed": [
        {
          "kind": "field",
          "name": "GoldenTestCase.MaxNumRunningReqs",
          "sig": "int64"
        },
        {
          "kind": "field",
          "name": "GoldenTestCase.MaxNumScheduledTokens",
          "sig": "int64"
        }
      ]
    }
  ],
  "components": {
    "module": "github.com/inference-sim/inference-sim",
    "depth": 2,
    "components": [
      {
        "name": "(root)",
        "members": [
          ""
        ],
        "inCycle": false
      },
      {
        "name": "cmd",
        "members": [
          "cmd"
        ],
        "inCycle": false,
        "change": "minor"
      },
      {
        "name": "sim",
        "members": [
          "sim"
        ],
        "inCycle": false,
        "change": "minor"
      },
      {
        "name": "sim/cluster",
        "members": [
          "sim/cluster"
        ],
        "inCycle": false,
        "change": "minor"
      },
      {
        "name": "sim/internal",
        "members": [
          "sim/internal/hash",
          "sim/internal/testutil",
          "sim/internal/tokenid",
          "sim/internal/util"
        ],
        "inCycle": false,
        "change": "minor"
      },
      {
        "name": "sim/kv",
        "members": [
          "sim/kv"
        ],
        "inCycle": false
      },
      {
        "name": "sim/latency",
        "members": [
          "sim/latency"
        ],
        "inCycle": false
      },
      {
        "name": "sim/lora",
        "members": [
          "sim/lora"
        ],
        "inCycle": false
      },
      {
        "name": "sim/saturation",
        "members": [
          "sim/saturation"
        ],
        "inCycle": false
      },
      {
        "name": "sim/trace",
        "members": [
          "sim/trace"
        ],
        "inCycle": false
      },
      {
        "name": "sim/workload",
        "members": [
          "sim/workload"
        ],
        "inCycle": false
      }
    ],
    "edges": [
      {
        "from": "(root)",
        "to": "cmd",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/cluster",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/latency",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/lora",
        "kind": "import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/saturation",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/trace",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "cmd",
        "to": "sim/workload",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim",
        "to": "sim/internal",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/cluster",
        "to": "sim",
        "kind": "call, implements, import",
        "change": ""
      },
      {
        "from": "sim/cluster",
        "to": "sim/kv",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/cluster",
        "to": "sim/latency",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/cluster",
        "to": "sim/trace",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/cluster",
        "to": "sim/workload",
        "kind": "import",
        "change": ""
      },
      {
        "from": "sim/kv",
        "to": "sim",
        "kind": "call, implements, import",
        "change": ""
      },
      {
        "from": "sim/kv",
        "to": "sim/internal",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/latency",
        "to": "sim",
        "kind": "call, implements, import",
        "change": ""
      },
      {
        "from": "sim/lora",
        "to": "sim",
        "kind": "implements, import",
        "change": ""
      },
      {
        "from": "sim/saturation",
        "to": "sim",
        "kind": "import",
        "change": ""
      },
      {
        "from": "sim/saturation",
        "to": "sim/workload",
        "kind": "call, import",
        "change": ""
      },
      {
        "from": "sim/workload",
        "to": "sim",
        "kind": "call, import",
        "change": ""
      }
    ]
  }
}
```

</details>

