# Autoscaler CLI Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Phase 1A NodePools and Phase 1C-1b autoscaler pipeline to `blis run` via `--policy-config` YAML extension, enabling a full scale-up/scale-down demo.

**Architecture:** Extend `PolicyBundle` in `sim/bundle.go` with parallel YAML types for node pools and autoscaler config (avoiding circular imports). `NewClusterSimulator` wires the default pipeline (`DefaultCollector + V2SaturationAnalyzer + UnlimitedEngine + DirectActuator`) using `DeploymentConfig.AutoscalerAnalyzerConfig`. `cmd/root.go` converts bundle fields to `DeploymentConfig` and exposes one CLI override flag.

**Tech Stack:** Go 1.22+, `gopkg.in/yaml.v3` (strict parsing), cobra, `sim/cluster` package

---

## File Map

| File | Change |
|------|--------|
| `sim/bundle.go` | Add `NodePoolBundleConfig`, `DelayBundleSpec`, `AnalyzerBundleConfig`, `AutoscalerBundleConfig`; add `NodePools` and `Autoscaler` fields to `PolicyBundle`; extend `Validate()` |
| `sim/bundle_test.go` | Tests for YAML round-trip, validation errors, zero-value safety |
| `sim/cluster/deployment.go` | Add `AutoscalerAnalyzerConfig V2SaturationAnalyzerConfig` field |
| `sim/cluster/cluster.go` | Replace nil-component pipeline init with wired default pipeline |
| `sim/cluster/cluster_test.go` (or new file) | Test: autoscaler wired when interval > 0, default thresholds applied |
| `cmd/root.go` | Add `--model-autoscaler-interval-us` flag; convert bundle → `DeploymentConfig` |
| `examples/autoscaler-demo.yaml` | New self-contained demo YAML |

---

### Task 1: Add YAML types and extend PolicyBundle

**Files:**
- Modify: `sim/bundle.go`
- Modify: `sim/bundle_test.go`

- [ ] **Step 1.1: Write failing tests for new YAML sections**

Add to `sim/bundle_test.go`:

```go
func TestLoadPolicyBundle_AutoscalerSection(t *testing.T) {
	yaml := `
autoscaler:
  interval_us: 30000000
  scale_up_cooldown_us: 60000000
  scale_down_cooldown_us: 180000000
  actuation_delay:
    mean: 10.0
    stddev: 2.0
  analyzer:
    kv_cache_threshold: 0.8
    scale_up_threshold: 0.8
    scale_down_boundary: 0.4
    avg_input_tokens: 512.0
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.Autoscaler.IntervalUs != 30_000_000 {
		t.Errorf("IntervalUs = %v, want 30000000", bundle.Autoscaler.IntervalUs)
	}
	if bundle.Autoscaler.Analyzer.KVCacheThreshold != 0.8 {
		t.Errorf("KVCacheThreshold = %v, want 0.8", bundle.Autoscaler.Analyzer.KVCacheThreshold)
	}
	if bundle.Autoscaler.ActuationDelay.Mean != 10.0 {
		t.Errorf("ActuationDelay.Mean = %v, want 10.0", bundle.Autoscaler.ActuationDelay.Mean)
	}
}

func TestLoadPolicyBundle_NodePoolsSection(t *testing.T) {
	yaml := `
node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80.0
    initial_nodes: 1
    min_nodes: 1
    max_nodes: 4
    cost_per_hour: 32.0
    provisioning_delay:
      mean: 30.0
      stddev: 5.0
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if len(bundle.NodePools) != 1 {
		t.Fatalf("NodePools len = %d, want 1", len(bundle.NodePools))
	}
	np := bundle.NodePools[0]
	if np.Name != "h100-pool" {
		t.Errorf("Name = %q, want h100-pool", np.Name)
	}
	if np.MaxNodes != 4 {
		t.Errorf("MaxNodes = %d, want 4", np.MaxNodes)
	}
	if np.ProvisioningDelay.Mean != 30.0 {
		t.Errorf("ProvisioningDelay.Mean = %v, want 30.0", np.ProvisioningDelay.Mean)
	}
}

func TestLoadPolicyBundle_AutoscalerAbsent_IsZero(t *testing.T) {
	// Existing policy-config files without autoscaler section must parse cleanly.
	yaml := `
admission:
  policy: always-admit
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.Autoscaler.IntervalUs != 0 {
		t.Errorf("IntervalUs = %v, want 0 (disabled)", bundle.Autoscaler.IntervalUs)
	}
	if len(bundle.NodePools) != 0 {
		t.Errorf("NodePools len = %d, want 0", len(bundle.NodePools))
	}
}

func TestPolicyBundle_Validate_AutoscalerNegativeInterval(t *testing.T) {
	bundle := &PolicyBundle{
		Autoscaler: AutoscalerBundleConfig{IntervalUs: -1},
	}
	if err := bundle.Validate(); err == nil {
		t.Error("expected error for negative interval_us, got nil")
	}
}

func TestPolicyBundle_Validate_NodePool_MissingName(t *testing.T) {
	bundle := &PolicyBundle{
		NodePools: []NodePoolBundleConfig{
			{GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, MaxNodes: 2},
		},
	}
	if err := bundle.Validate(); err == nil {
		t.Error("expected error for missing node pool name, got nil")
	}
}
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /path/to/worktree  # .worktrees/feat-autoscaler-cli
go test ./sim/... -run "TestLoadPolicyBundle_Autoscaler|TestLoadPolicyBundle_NodePool|TestPolicyBundle_Validate_Autoscaler|TestPolicyBundle_Validate_NodePool" -v 2>&1 | tail -20
```

Expected: compilation errors (types not defined yet).

- [ ] **Step 1.3: Add new types and fields to sim/bundle.go**

After the existing `PriorityConfig` struct (around line 44), add:

```go
// NodePoolBundleConfig mirrors cluster.NodePoolConfig for YAML loading in PolicyBundle.
// Converted to cluster.NodePoolConfig in cmd/ to avoid a sim→sim/cluster circular import.
type NodePoolBundleConfig struct {
	Name              string          `yaml:"name"`
	GPUType           string          `yaml:"gpu_type"`
	GPUsPerNode       int             `yaml:"gpus_per_node"`
	GPUMemoryGiB      float64         `yaml:"gpu_memory_gib"`
	InitialNodes      int             `yaml:"initial_nodes"`
	MinNodes          int             `yaml:"min_nodes"`
	MaxNodes          int             `yaml:"max_nodes"`
	ProvisioningDelay DelayBundleSpec `yaml:"provisioning_delay"`
	CostPerHour       float64         `yaml:"cost_per_hour"`
}

// DelayBundleSpec mirrors cluster.DelaySpec for YAML loading. Mean and Stddev in seconds.
type DelayBundleSpec struct {
	Mean   float64 `yaml:"mean"`
	Stddev float64 `yaml:"stddev"`
}

// AnalyzerBundleConfig holds V2SaturationAnalyzer thresholds.
// Zero values signal "use defaults" (filled by effectiveAnalyzerConfig in cluster package).
type AnalyzerBundleConfig struct {
	KVCacheThreshold  float64 `yaml:"kv_cache_threshold"`
	ScaleUpThreshold  float64 `yaml:"scale_up_threshold"`
	ScaleDownBoundary float64 `yaml:"scale_down_boundary"`
	AvgInputTokens    float64 `yaml:"avg_input_tokens"`
}

// AutoscalerBundleConfig holds autoscaler pipeline configuration.
// IntervalUs == 0 means the autoscaler is disabled (default).
type AutoscalerBundleConfig struct {
	IntervalUs          float64              `yaml:"interval_us"`
	ScaleUpCooldownUs   float64              `yaml:"scale_up_cooldown_us"`
	ScaleDownCooldownUs float64              `yaml:"scale_down_cooldown_us"`
	ActuationDelay      DelayBundleSpec      `yaml:"actuation_delay"`
	Analyzer            AnalyzerBundleConfig `yaml:"analyzer"`
}
```

Add two fields to `PolicyBundle`:

```go
type PolicyBundle struct {
	Admission     AdmissionConfig       `yaml:"admission"`
	Routing       RoutingConfig         `yaml:"routing"`
	Priority      PriorityConfig        `yaml:"priority"`
	Scheduler     string                `yaml:"scheduler"`
	TenantBudgets map[string]float64    `yaml:"tenant_budgets"`
	NodePools     []NodePoolBundleConfig `yaml:"node_pools"`
	Autoscaler    AutoscalerBundleConfig `yaml:"autoscaler"`
}
```

- [ ] **Step 1.4: Extend Validate() for new fields**

Add at the end of `Validate()` before the final `return nil`:

```go
	// Validate autoscaler config when enabled.
	if math.IsNaN(b.Autoscaler.IntervalUs) || math.IsInf(b.Autoscaler.IntervalUs, 0) {
		return fmt.Errorf("autoscaler.interval_us must be a finite number")
	}
	if b.Autoscaler.IntervalUs < 0 {
		return fmt.Errorf("autoscaler.interval_us must be >= 0 (0 = disabled), got %v", b.Autoscaler.IntervalUs)
	}
	if b.Autoscaler.ScaleUpCooldownUs < 0 {
		return fmt.Errorf("autoscaler.scale_up_cooldown_us must be >= 0, got %v", b.Autoscaler.ScaleUpCooldownUs)
	}
	if b.Autoscaler.ScaleDownCooldownUs < 0 {
		return fmt.Errorf("autoscaler.scale_down_cooldown_us must be >= 0, got %v", b.Autoscaler.ScaleDownCooldownUs)
	}
	if b.Autoscaler.ActuationDelay.Mean < 0 {
		return fmt.Errorf("autoscaler.actuation_delay.mean must be >= 0, got %v", b.Autoscaler.ActuationDelay.Mean)
	}
	if b.Autoscaler.ActuationDelay.Stddev < 0 {
		return fmt.Errorf("autoscaler.actuation_delay.stddev must be >= 0, got %v", b.Autoscaler.ActuationDelay.Stddev)
	}
	// Validate node pools: name and gpu_type required; gpus_per_node >= 1; gpu_memory_gib > 0; max_nodes >= initial_nodes.
	for i, np := range b.NodePools {
		if np.Name == "" {
			return fmt.Errorf("node_pools[%d]: name must not be empty", i)
		}
		if np.GPUType == "" {
			return fmt.Errorf("node_pools[%d] %q: gpu_type must not be empty", i, np.Name)
		}
		if np.GPUsPerNode < 1 {
			return fmt.Errorf("node_pools[%d] %q: gpus_per_node must be >= 1, got %d", i, np.Name, np.GPUsPerNode)
		}
		if np.GPUMemoryGiB <= 0 {
			return fmt.Errorf("node_pools[%d] %q: gpu_memory_gib must be > 0, got %v", i, np.Name, np.GPUMemoryGiB)
		}
		if np.MaxNodes < np.InitialNodes {
			return fmt.Errorf("node_pools[%d] %q: max_nodes (%d) must be >= initial_nodes (%d)", i, np.Name, np.MaxNodes, np.InitialNodes)
		}
	}
```

- [ ] **Step 1.5: Run tests**

```bash
go test ./sim/... -run "TestLoadPolicyBundle_Autoscaler|TestLoadPolicyBundle_NodePool|TestPolicyBundle_Validate_Autoscaler|TestPolicyBundle_Validate_NodePool" -v 2>&1 | tail -20
```

Expected: all 5 new tests pass.

- [ ] **Step 1.6: Run full sim suite to check no regressions**

```bash
go test ./sim/... 2>&1 | tail -5
```

Expected: all existing tests still pass.

- [ ] **Step 1.7: Commit**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(bundle): add node_pools and autoscaler sections to PolicyBundle"
```

---

### Task 2: Add AutoscalerAnalyzerConfig to DeploymentConfig and wire default pipeline

**Files:**
- Modify: `sim/cluster/deployment.go`
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/cluster_test.go` (add new test)

- [ ] **Step 2.1: Write failing test**

Add to `sim/cluster/cluster_test.go` (find the block of `TestNewClusterSimulator` tests):

```go
func TestNewClusterSimulator_AutoscalerWiredWhenEnabled(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.ModelAutoscalerIntervalUs = 30_000_000
	// AutoscalerAnalyzerConfig zero values → defaults applied inside constructor
	cs := NewClusterSimulator(cfg, nil, nil)
	if cs.autoscaler == nil {
		t.Fatal("autoscaler must not be nil when ModelAutoscalerIntervalUs > 0")
	}
	if cs.autoscaler.collector == nil {
		t.Error("autoscaler.collector must not be nil")
	}
	if cs.autoscaler.analyzer == nil {
		t.Error("autoscaler.analyzer must not be nil")
	}
	if cs.autoscaler.engine == nil {
		t.Error("autoscaler.engine must not be nil")
	}
	if cs.autoscaler.actuator == nil {
		t.Error("autoscaler.actuator must not be nil")
	}
}

func TestNewClusterSimulator_AutoscalerNilWhenDisabled(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	// ModelAutoscalerIntervalUs == 0 (default) → autoscaler stays nil
	cs := NewClusterSimulator(cfg, nil, nil)
	if cs.autoscaler != nil {
		t.Error("autoscaler must be nil when ModelAutoscalerIntervalUs == 0")
	}
}
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
go test ./sim/cluster/... -run "TestNewClusterSimulator_Autoscaler" -v 2>&1 | tail -15
```

Expected: `TestNewClusterSimulator_AutoscalerWiredWhenEnabled` fails — `autoscaler.collector` is nil.

- [ ] **Step 2.3: Add AutoscalerAnalyzerConfig field to DeploymentConfig**

In `sim/cluster/deployment.go`, add after `ScaleDownCooldownUs`:

```go
// AutoscalerAnalyzerConfig holds V2SaturationAnalyzer thresholds.
// Zero values are safe: NewClusterSimulator applies defaults (KvCacheThreshold=0.8,
// ScaleUpThreshold=0.8, ScaleDownBoundary=0.4, AvgInputTokens=512).
AutoscalerAnalyzerConfig V2SaturationAnalyzerConfig `yaml:"autoscaler_analyzer,omitempty"`
```

- [ ] **Step 2.4: Add effectiveAnalyzerConfig helper to cluster.go**

Add the following function to `sim/cluster/cluster.go`, just before `NewClusterSimulator`:

```go
// effectiveAnalyzerConfig applies defaults to a V2SaturationAnalyzerConfig.
// Zero values mean "not configured" — fill with WVA reference defaults.
// Keeps callers (cmd/, tests) simple: setting interval_us is enough to enable
// the autoscaler with sensible thresholds.
func effectiveAnalyzerConfig(cfg V2SaturationAnalyzerConfig) V2SaturationAnalyzerConfig {
	if cfg.KvCacheThreshold == 0 {
		cfg.KvCacheThreshold = 0.8
	}
	if cfg.ScaleUpThreshold == 0 {
		cfg.ScaleUpThreshold = 0.8
	}
	if cfg.ScaleDownBoundary == 0 {
		cfg.ScaleDownBoundary = 0.4
	}
	if cfg.AvgInputTokens == 0 {
		cfg.AvgInputTokens = 512
	}
	return cfg
}
```

- [ ] **Step 2.5: Wire default pipeline in NewClusterSimulator**

In `sim/cluster/cluster.go`, replace the autoscaler init block (currently lines 347–351):

```go
	if config.ModelAutoscalerIntervalUs > 0 {
		// Interface components (collector, analyzer, engine, actuator) are injected by
		// tests or cmd/ after construction, before Run() is called (see newAutoscalerPipeline).
		cs.autoscaler = newAutoscalerPipeline(nil, nil, nil, nil, rng.ForSubsystem(subsystemAutoscaler))
	}
```

Replace with:

```go
	if config.ModelAutoscalerIntervalUs > 0 {
		// Wire the default WVA pipeline: DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator.
		// effectiveAnalyzerConfig fills zero fields with WVA reference defaults so callers only need to set interval_us.
		// Tests that need custom components (stubs, nopActuator) replace cs.autoscaler after construction (same-package access).
		analyzerCfg := effectiveAnalyzerConfig(config.AutoscalerAnalyzerConfig)
		cs.autoscaler = newAutoscalerPipeline(
			&DefaultCollector{},
			NewV2SaturationAnalyzer(analyzerCfg),
			&UnlimitedEngine{},
			NewDirectActuator(cs),
			rng.ForSubsystem(subsystemAutoscaler),
		)
	}
```

- [ ] **Step 2.6: Run the new tests**

```bash
go test ./sim/cluster/... -run "TestNewClusterSimulator_Autoscaler" -v 2>&1 | tail -15
```

Expected: both tests pass.

- [ ] **Step 2.7: Run full cluster suite**

```bash
go test ./sim/cluster/... 2>&1 | tail -5
```

Expected: all existing tests still pass (tests that replaced `cs.autoscaler` with stubs still work — they overwrite after construction).

- [ ] **Step 2.8: Commit**

```bash
git add sim/cluster/deployment.go sim/cluster/cluster.go sim/cluster/cluster_test.go
git commit -m "feat(cluster): wire default autoscaler pipeline in NewClusterSimulator"
```

---

### Task 3: Wire CLI flag and bundle conversion in cmd/root.go

**Files:**
- Modify: `cmd/root.go`
- Modify: `cmd/default_config_test.go` (or appropriate cmd test file)

- [ ] **Step 3.1: Write failing CLI test**

Find `cmd/default_config_test.go` and add a test (or look for appropriate existing test file — there is a `simconfig_shared_test.go`). Add to `cmd/root_test.go` or the most appropriate file:

```go
func TestRunCmd_AutoscalerFlagSetsConfig(t *testing.T) {
	// Verify --model-autoscaler-interval-us flag is registered and accepted.
	cmd := newRunCmd()
	if err := cmd.ParseFlags([]string{"--model-autoscaler-interval-us", "30000000"}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	v, err := cmd.Flags().GetFloat64("model-autoscaler-interval-us")
	if err != nil {
		t.Fatalf("GetFloat64: %v", err)
	}
	if v != 30_000_000 {
		t.Errorf("model-autoscaler-interval-us = %v, want 30000000", v)
	}
}
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
go test ./cmd/... -run "TestRunCmd_AutoscalerFlagSetsConfig" -v 2>&1 | tail -10
```

Expected: FAIL — flag not registered.

- [ ] **Step 3.3: Add the flag declaration in cmd/root.go**

Find the block where `snapshotRefreshInterval` and related flags are declared (around line 975). Add after it:

```go
var modelAutoscalerIntervalUs float64
cmd.Flags().Float64Var(&modelAutoscalerIntervalUs, "model-autoscaler-interval-us", 0,
	"Autoscaler tick interval in microseconds (0 = disabled). Overrides policy-config autoscaler.interval_us when non-zero.")
```

- [ ] **Step 3.4: Add bundle → DeploymentConfig conversion for autoscaler and node pools**

In the `if policyConfigPath != ""` block (around line 800–839), add after the existing bundle field applications:

```go
		// Apply autoscaler config from bundle (CLI flag overrides below).
		if bundle.Autoscaler.IntervalUs > 0 {
			config_autoscalerIntervalUs = bundle.Autoscaler.IntervalUs
			config_scaleUpCooldownUs = bundle.Autoscaler.ScaleUpCooldownUs
			config_scaleDownCooldownUs = bundle.Autoscaler.ScaleDownCooldownUs
			config_actuationDelayMean = bundle.Autoscaler.ActuationDelay.Mean
			config_actuationDelayStddev = bundle.Autoscaler.ActuationDelay.Stddev
			config_analyzerCfg = cluster.V2SaturationAnalyzerConfig{
				KvCacheThreshold:  bundle.Autoscaler.Analyzer.KVCacheThreshold,
				ScaleUpThreshold:  bundle.Autoscaler.Analyzer.ScaleUpThreshold,
				ScaleDownBoundary: bundle.Autoscaler.Analyzer.ScaleDownBoundary,
				AvgInputTokens:    bundle.Autoscaler.Analyzer.AvgInputTokens,
			}
		}
		// Convert node pools from bundle to cluster.NodePoolConfig.
		for _, np := range bundle.NodePools {
			bundleNodePools = append(bundleNodePools, cluster.NodePoolConfig{
				Name:         np.Name,
				GPUType:      np.GPUType,
				GPUsPerNode:  np.GPUsPerNode,
				GPUMemoryGiB: np.GPUMemoryGiB,
				InitialNodes: np.InitialNodes,
				MinNodes:     np.MinNodes,
				MaxNodes:     np.MaxNodes,
				ProvisioningDelay: cluster.DelaySpec{
					Mean:   np.ProvisioningDelay.Mean,
					Stddev: np.ProvisioningDelay.Stddev,
				},
				CostPerHour: np.CostPerHour,
			})
		}
```

Declare the new local variables before the `if policyConfigPath != ""` block (alongside the existing variable declarations in the `RunE` closure):

```go
var (
	config_autoscalerIntervalUs  float64
	config_scaleUpCooldownUs     float64
	config_scaleDownCooldownUs   float64
	config_actuationDelayMean    float64
	config_actuationDelayStddev  float64
	config_analyzerCfg           cluster.V2SaturationAnalyzerConfig
	bundleNodePools              []cluster.NodePoolConfig
)
```

After the bundle block, apply the CLI flag override:

```go
// CLI flag --model-autoscaler-interval-us overrides YAML autoscaler.interval_us when non-zero.
if cmd.Flags().Changed("model-autoscaler-interval-us") && modelAutoscalerIntervalUs > 0 {
	config_autoscalerIntervalUs = modelAutoscalerIntervalUs
}
```

- [ ] **Step 3.5: Wire the new variables into DeploymentConfig**

In the `config := cluster.DeploymentConfig{...}` block (around line 1509), add the new fields:

```go
		ModelAutoscalerIntervalUs:  config_autoscalerIntervalUs,
		ScaleUpCooldownUs:          config_scaleUpCooldownUs,
		ScaleDownCooldownUs:        config_scaleDownCooldownUs,
		ActuationDelay:             cluster.DelaySpec{Mean: config_actuationDelayMean, Stddev: config_actuationDelayStddev},
		AutoscalerAnalyzerConfig:   config_analyzerCfg,
		NodePools:                  bundleNodePools,
```

- [ ] **Step 3.6: Build to verify no compilation errors**

```bash
go build ./... 2>&1
```

Expected: clean build.

- [ ] **Step 3.7: Run the flag test**

```bash
go test ./cmd/... -run "TestRunCmd_AutoscalerFlagSetsConfig" -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3.8: Run full cmd suite**

```bash
go test ./cmd/... 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 3.9: Commit**

```bash
git add cmd/root.go cmd/root_test.go
git commit -m "feat(cmd): wire autoscaler CLI flag and policy-config bundle conversion"
```

---

### Task 4: Create examples/autoscaler-demo.yaml and smoke test

**Files:**
- Create: `examples/autoscaler-demo.yaml`

- [ ] **Step 4.1: Create the demo YAML**

Create `examples/autoscaler-demo.yaml`:

```yaml
# BLIS Autoscaler Demo — Scale Up and Scale Down
#
# Demonstrates the Phase 1C-1b model autoscaler (WVA-aligned pipeline):
#   DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator
#
# The autoscaler fires every 30s, observes KV utilization + queue depth,
# and scales up when demand exceeds capacity (util > 0.8) or scales down
# when capacity is idle (util < 0.4 per replica after removal).
#
# Node pool provides 3 spare instances beyond --num-instances=2:
#   initial: 2 instances (from --num-instances)
#   max:     up to 5 total (pool has max_nodes=2 which adds 2 more GPU nodes,
#            each holding 1 TP=1 instance at the roofline/blackbox GPU ratio)
#
# ============================================================================
# TRY IT
# ============================================================================
#
#   # Build first
#   go build -o blis main.go
#
#   # Baseline: 2 instances, no autoscaler
#   ./blis run \
#     --model meta-llama/Llama-2-7b-hf \
#     --latency-model blackbox \
#     --num-instances 2 \
#     --rate 40 --num-requests 400 \
#     --horizon 300000000
#
#   # With autoscaler: starts with 2, scales up under load, scales down when quiet
#   ./blis run \
#     --model meta-llama/Llama-2-7b-hf \
#     --latency-model blackbox \
#     --num-instances 2 \
#     --policy-config examples/autoscaler-demo.yaml \
#     --workload-spec examples/regression_workload_load_spikes.yaml \
#     --rate 40 --num-requests 400 \
#     --horizon 300000000
#
# ============================================================================
# What to observe
# ============================================================================
#
#   Look for "[actuator] scale-up" and "[actuator] scale-down" log lines.
#   With a bursty workload, you should see:
#     - Scale-up triggered when the spike hits and queues grow
#     - Scale-down triggered after traffic subsides and capacity is idle
#
# ============================================================================

admission:
  policy: always-admit

routing:
  policy: weighted
  scorers:
    - name: queue-depth
      weight: 2.0
    - name: kv-utilization
      weight: 1.0

# Node pool: provides spare GPU capacity the autoscaler can grow into.
# initial_nodes=1 means 1 extra node pre-provisioned (available immediately).
# max_nodes=3 means up to 3 nodes total can be provisioned (3 extra instances).
# cost_per_hour=2.0 is used by the Engine to prefer cheaper variants.
node_pools:
  - name: demo-pool
    gpu_type: A100       # matches --hardware (default A100 for blackbox backend)
    gpus_per_node: 1     # 1 GPU per node → 1 instance per node at TP=1
    gpu_memory_gib: 40.0
    initial_nodes: 1     # 1 node ready immediately (fast scale-up path)
    min_nodes: 0
    max_nodes: 3         # headroom: up to 3 additional instances
    cost_per_hour: 2.0
    provisioning_delay:
      mean: 0.0          # 0s delay for demo clarity (set to 30.0 for realistic lag)
      stddev: 0.0

# Autoscaler pipeline configuration.
# interval_us: tick every 30s of simulated time
# scale_up_cooldown_us: wait 60s between scale-up decisions (avoid flapping)
# scale_down_cooldown_us: wait 180s before scaling down (conservative)
# actuation_delay: 10s mean lag modeling HPA/KEDA scrape latency
autoscaler:
  interval_us: 30000000
  scale_up_cooldown_us: 60000000
  scale_down_cooldown_us: 180000000
  actuation_delay:
    mean: 10.0
    stddev: 2.0
  analyzer:
    kv_cache_threshold: 0.8    # use 80% of KV capacity as effective supply
    scale_up_threshold: 0.8    # scale up when demand > 80% of supply
    scale_down_boundary: 0.4   # scale down when spare capacity > 60% of supply
    avg_input_tokens: 512.0    # queue depth → token demand conversion
```

- [ ] **Step 4.2: Smoke test — build and run with demo config**

Build and verify the demo runs without crashing. Use the blackbox latency model (no HF config needed):

```bash
go build -o blis main.go 2>&1 && echo "BUILD OK"

./blis run \
  --model meta-llama/Llama-2-7b-hf \
  --latency-model blackbox \
  --num-instances 2 \
  --policy-config examples/autoscaler-demo.yaml \
  --workload-spec examples/regression_workload_load_spikes.yaml \
  --rate 40 --num-requests 400 \
  --horizon 300000000 \
  --log-level info 2>&1 | grep -E "autoscal|scale.up|scale.down|actuator|completed_requests" | head -30
```

Expected: output includes autoscaler tick log lines and no panics. Final line includes `completed_requests`.

- [ ] **Step 4.3: Run full test suite**

```bash
go test ./... 2>&1 | tail -15
```

Expected: all packages pass.

- [ ] **Step 4.4: Commit**

```bash
git add examples/autoscaler-demo.yaml
git commit -m "feat(examples): add autoscaler-demo.yaml with node pool and WVA pipeline config"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `node_pools:` + `autoscaler:` YAML in PolicyBundle | Task 1 |
| Parallel bundle types avoid circular import | Task 1 (NodePoolBundleConfig, DelayBundleSpec in sim/) |
| Pipeline wired inside NewClusterSimulator | Task 2 |
| effectiveAnalyzerConfig defaults | Task 2 (Step 2.4) |
| `--model-autoscaler-interval-us` CLI flag | Task 3 |
| Bundle → DeploymentConfig conversion | Task 3 |
| `examples/autoscaler-demo.yaml` | Task 4 |
| INV-6: existing runs byte-identical (interval=0 = no change) | Task 2 Step 2.7 (existing tests pass) |

**Placeholder scan:** No TBDs or "implement later" — all code blocks are complete.

**Type consistency:**
- `AutoscalerBundleConfig.IntervalUs` used consistently in Tasks 1 and 3
- `V2SaturationAnalyzerConfig` field names (`KvCacheThreshold`, `ScaleUpThreshold`, `ScaleDownBoundary`, `AvgInputTokens`) match `saturation_analyzer.go`
- `cluster.DelaySpec{Mean, Stddev}` field names match `infra_config.go`
- `config_autoscalerIntervalUs` local var in Task 3 maps to `DeploymentConfig.ModelAutoscalerIntervalUs`

**Note for Task 3:** The exact line numbers in `cmd/root.go` (e.g., line 975, 800, 1509) may shift as the file is edited. Search by the surrounding code patterns shown in the steps rather than relying on line numbers.
