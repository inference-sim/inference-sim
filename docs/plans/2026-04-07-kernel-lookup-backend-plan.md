# Kernel-Lookup Latency Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new latency backend that uses aiconfigurator's measured kernel latencies as basis functions, with learned γ corrections, to predict step time more accurately than the analytical roofline.

**Architecture:** Offline Python script queries aiconfigurator DB to produce per-model `kernel_profile.yaml` with 1D/2D lookup tables (per-layer, token-count-indexed). Go runtime loads these tables and computes step time via γ-weighted interpolated lookups.

**Formula:**
```
StepTime = γ₁·T_pf_gemm + γ₂·T_pf_attn + γ₃·T_dc_gemm + γ₄·T_dc_attn
         + γ₆·T_allreduce + γ₇·T_moe + γ₈·numLayers + γ₉·batchSize + γ₁₀
```
Note: γ₅ (weight loading) is dropped — it double-counts with GEMM measurements.

**Basis function conventions (must be consistent between Python and Go):**
- All lookup tables store **per-layer latencies** (Python script divides by `numLayers`)
- Context attention table axes: `(batch_size, avg_isl)` — matches aiconfigurator `(b, s)` signature
- Generation attention table axes: `(decode_tokens, avg_context)` — decode_tokens == decode_batch_size
- AllReduce table axis: **token count** (Python script converts `message_size → tokens`)
- AllReduce multiplier: `2·numDenseLayers + 1·numMoELayers` (not `numLayers`)
- MoE table covers expert FFN only (not attention-projection GEMMs)

**Tech Stack:** Go 1.22+, Python 3.10+, aiconfigurator SDK, gopkg.in/yaml.v3

**Design doc:** `docs/plans/2026-04-07-kernel-lookup-backend-design.md`

---

## Audit Fixes Applied

These bugs were found during design review and are corrected throughout:

| Bug | Fix |
|-----|-----|
| AllReduce keyed by message_size, interpolated by tokens | YAML stores token-count axis; Python converts msg_size→tokens |
| AllReduce `* L` overcounts (should be 2·dense + 1·MoE) | Use `allReduceUnits = 2·numDenseLayers + numMoELayers` |
| Context attention uses totalPrefillTokens (wrong) | Use `numPrefillRequests` for batch axis (matches aiconfigurator `b`) |
| Factory dead-end: NewKernelLookupModel always errors | Add `KernelProfilePath` field to `ModelHardwareConfig` |
| T_weight double-counts GEMM memory access | Removed from formula (set γ₅=0, drop term) |
| Per-layer vs per-forward-pass table ambiguity | Tables are per-layer; Python divides by numLayers; documented in schema |

---

### Task 1: Register "kernel-lookup" as Valid Backend + Extend ModelHardwareConfig

**Files:**
- Modify: `sim/bundle.go:69`
- Modify: `sim/config.go:98-125` (`ModelHardwareConfig` + constructor)
- Modify: `sim/latency/latency.go:131-284`

**Step 1: Add "kernel-lookup" to validLatencyBackends**

In `sim/bundle.go:69`:

```go
validLatencyBackends = map[string]bool{
    "": true, "blackbox": true, "roofline": true, "crossmodel": true,
    "trained-roofline": true, "evolved": true, "kernel-lookup": true,
}
```

**Step 2: Add `KernelProfilePath` to `ModelHardwareConfig`**

In `sim/config.go:98-125`, the existing struct and constructor:

```go
type ModelHardwareConfig struct {
    ModelConfig     ModelConfig
    HWConfig        HardwareCalib
    Model           string
    GPU             string
    TP              int
    Backend         string
    MaxModelLen     int64
    KernelProfilePath string // path to kernel_profile.yaml; only used by Backend="kernel-lookup"
}
```

Update `NewModelHardwareConfig` to keep the same signature for existing callers (add the field as an optional field set by a new helper, not in the constructor signature — to avoid touching 143 call sites). Instead of changing the constructor signature, add a setter method:

```go
// WithKernelProfilePath returns a copy with KernelProfilePath set.
// Used by the CLI when constructing a kernel-lookup model config.
func (hw ModelHardwareConfig) WithKernelProfilePath(path string) ModelHardwareConfig {
    hw.KernelProfilePath = path
    return hw
}
```

This is backward-compatible: all existing callers get `KernelProfilePath=""` from the zero value.

**Step 3: Write the test for WithKernelProfilePath**

In `sim/config_test.go`, add:

```go
func TestModelHardwareConfig_WithKernelProfilePath(t *testing.T) {
    base := NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0)
    assert.Equal(t, "", base.KernelProfilePath)
    with := base.WithKernelProfilePath("/tmp/profile.yaml")
    assert.Equal(t, "/tmp/profile.yaml", with.KernelProfilePath)
    // Original is unchanged (value receiver)
    assert.Equal(t, "", base.KernelProfilePath)
}
```

Run: `go test ./sim/... -run TestModelHardwareConfig_WithKernelProfilePath -v`
Expected: FAIL (method not defined yet), then PASS after implementation.

**Step 4: Add factory stub in `latency.go`**

In `sim/latency/latency.go`, add before `default:`:

```go
case "kernel-lookup":
    return NewKernelLookupModel(coeffs, hw)
```

This calls a function that will be implemented in Task 3. It won't compile until then — that's the build-break that signals we must complete Task 3.

**Step 5: Run tests to verify registration**

Run: `go build ./...`
Expected: FAIL with "undefined: NewKernelLookupModel" — intentional, resolved in Task 3.

Run: `go test ./sim/... -run TestValidLatencyBackendNames -v`
Expected: PASS (registration is in bundle.go, no compile needed for that test).

**Step 6: Commit**

```bash
git add sim/bundle.go sim/config.go sim/config_test.go sim/latency/latency.go
git commit -m "feat(latency): register kernel-lookup backend; add KernelProfilePath to ModelHardwareConfig"
```

---

### Task 2: Implement Lookup Table Types and YAML Parsing

**Files:**
- Create: `sim/latency/kernel_profile.go`
- Create: `sim/latency/kernel_profile_test.go`

**YAML schema contract (enforced by validation):**
- `context_gemm.batch_size` + `context_gemm.isl` → 2D context attention
- `context_gemm.tokens` → 1D GEMM (token count = total prefill tokens)
- `generation_gemm.tokens` → 1D GEMM (decode token count = decode batch size)
- `allreduce.tokens` → 1D, token-count-indexed (pre-converted from message_size by Python script)
- `allreduce.tp_size` → must match runtime TP (validated at construction)
- All latency values: **per-layer microseconds**
- `tp` field: tensor parallelism degree at which measurements were taken

**Step 1: Write the failing tests**

Create `sim/latency/kernel_profile_test.go`:

```go
package latency

import (
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

const testProfileYAML = `
gpu: h100_sxm
backend: vllm
version: "0.14.0"
model: meta-llama/Llama-2-7b-hf
tp: 1
num_layers: 32
num_moe_layers: 0
num_dense_layers: 32
hidden_dim: 4096

context_gemm:
  tokens: [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]
  latency_us: [1.56, 1.58, 1.65, 2.06, 5.31, 16.9, 56.7]

context_attention:
  batch_size: [1.0, 4.0, 16.0, 64.0]
  isl: [128.0, 512.0, 2048.0, 4096.0]
  latency_us:
    - [0.31, 0.38, 0.54, 0.95]
    - [0.33, 0.42, 0.68, 1.38]
    - [0.41, 0.62, 1.24, 2.48]
    - [0.65, 1.14, 2.48, 5.01]

generation_gemm:
  tokens: [1.0, 4.0, 16.0, 64.0, 256.0]
  latency_us: [1.50, 1.51, 1.55, 1.94, 5.01]

generation_attention:
  tokens: [1.0, 4.0, 16.0, 64.0, 256.0]
  context: [128.0, 512.0, 2048.0, 4096.0]
  latency_us:
    - [0.25, 0.31, 0.46, 0.84]
    - [0.26, 0.33, 0.52, 1.04]
    - [0.31, 0.46, 0.98, 1.96]
    - [0.50, 0.87, 1.87, 3.72]

allreduce:
  tokens: [1.0, 16.0, 256.0, 4096.0]
  tp_size: 1
  latency_us: [0.0, 0.0, 0.0, 0.0]
`

func writeTestProfile(t *testing.T, content string) string {
    t.Helper()
    dir := t.TempDir()
    path := filepath.Join(dir, "profile.yaml")
    require.NoError(t, os.WriteFile(path, []byte(content), 0644))
    return path
}

func TestLoadKernelProfile_ValidYAML(t *testing.T) {
    path := writeTestProfile(t, testProfileYAML)
    p, err := LoadKernelProfile(path)
    require.NoError(t, err)
    assert.Equal(t, "h100_sxm", p.GPU)
    assert.Equal(t, 1, p.TP)
    assert.Equal(t, 32, p.NumLayers)
    assert.Equal(t, 7, len(p.ContextGemm.Tokens))
    assert.Equal(t, 4, len(p.ContextAttention.BatchSize))
    assert.Equal(t, 4, len(p.ContextAttention.ISL))
    assert.Equal(t, 4, len(p.ContextAttention.LatencyUs))
    assert.Equal(t, 4, len(p.ContextAttention.LatencyUs[0]))
    assert.Nil(t, p.MoECompute) // dense model
}

func TestLoadKernelProfile_FileNotFound(t *testing.T) {
    _, err := LoadKernelProfile("/nonexistent/profile.yaml")
    assert.Error(t, err)
}

func TestLoadKernelProfile_InvalidTP(t *testing.T) {
    bad := `gpu: h100_sxm
tp: 0
num_layers: 32
context_gemm:
  tokens: [1.0]
  latency_us: [1.0]
context_attention:
  batch_size: [1.0]
  isl: [128.0]
  latency_us: [[1.0]]
generation_gemm:
  tokens: [1.0]
  latency_us: [1.0]
generation_attention:
  tokens: [1.0]
  context: [128.0]
  latency_us: [[1.0]]
allreduce:
  tokens: [1.0]
  tp_size: 1
  latency_us: [0.0]
`
    path := writeTestProfile(t, bad)
    _, err := LoadKernelProfile(path)
    assert.Error(t, err, "tp=0 should be rejected")
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/latency/... -run TestLoadKernelProfile -v`
Expected: FAIL with "undefined: LoadKernelProfile"

**Step 3: Implement `kernel_profile.go`**

Create `sim/latency/kernel_profile.go`:

```go
package latency

import (
    "fmt"
    "math"
    "os"
    "sort"

    "gopkg.in/yaml.v3"
)

// KernelProfile holds pre-computed per-layer latency lookup tables derived from
// aiconfigurator's measured GPU kernel database. All latency values are per-layer
// microseconds. Loaded from kernel_profile.yaml (generated offline by Python script).
//
// Conventions:
//   - context_gemm.tokens: total prefill tokens across batch
//   - context_attention.batch_size × context_attention.isl: prefill batch_size and avg ISL
//   - generation_gemm.tokens: decode token count (== decode batch size, 1 token/request)
//   - generation_attention.tokens × context: decode batch size and avg context length
//   - allreduce.tokens: total token count (Python script converts message_size → tokens)
//   - moe_compute.tokens: total token count (nil for dense models)
type KernelProfile struct {
    GPU            string `yaml:"gpu"`
    Backend        string `yaml:"backend"`
    Version        string `yaml:"version"`
    Model          string `yaml:"model"`
    TP             int    `yaml:"tp"`
    NumLayers      int    `yaml:"num_layers"`
    NumMoELayers   int    `yaml:"num_moe_layers"`
    NumDenseLayers int    `yaml:"num_dense_layers"`
    HiddenDim      int    `yaml:"hidden_dim"`

    ContextGemm         Lookup1D  `yaml:"context_gemm"`
    ContextAttention    Lookup2D  `yaml:"context_attention"`
    GenerationGemm      Lookup1D  `yaml:"generation_gemm"`
    GenerationAttention Lookup2D  `yaml:"generation_attention"`
    AllReduce           Lookup1D  `yaml:"allreduce"`
    MoECompute          *Lookup1D `yaml:"moe_compute,omitempty"`
}

// Lookup1D maps a single dimension (token count or message count) to per-layer latency (µs).
// Values must be sorted ascending and non-empty.
type Lookup1D struct {
    Tokens    []float64 `yaml:"tokens,omitempty"`
    LatencyUs []float64 `yaml:"latency_us"`
}

// Lookup2D maps two dimensions to per-layer latency (µs).
// For context_attention: BatchSize (primary) × ISL (secondary).
// For generation_attention: Tokens (primary) × Context (secondary).
// LatencyUs[i][j] = latency at secondary[i], primary[j].
type Lookup2D struct {
    BatchSize []float64   `yaml:"batch_size,omitempty"` // for context_attention
    ISL       []float64   `yaml:"isl,omitempty"`        // for context_attention
    Tokens    []float64   `yaml:"tokens,omitempty"`     // for generation_attention
    Context   []float64   `yaml:"context,omitempty"`    // for generation_attention
    LatencyUs [][]float64 `yaml:"latency_us"`
}

// AllReduceLookup extends Lookup1D with a TP size validation field.
type allReduceLookup struct {
    Lookup1D `yaml:",inline"`
    TPSize   int `yaml:"tp_size"`
}

// LoadKernelProfile reads and validates a kernel_profile.yaml file.
func LoadKernelProfile(path string) (*KernelProfile, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("load kernel profile %q: %w", path, err)
    }
    var p KernelProfile
    dec := yaml.NewDecoder(strings.NewReader(string(data)))
    dec.KnownFields(true)
    if err := dec.Decode(&p); err != nil {
        return nil, fmt.Errorf("parse kernel profile %q: %w", path, err)
    }
    if err := p.validate(); err != nil {
        return nil, fmt.Errorf("kernel profile %q invalid: %w", path, err)
    }
    return &p, nil
}
```

Wait — the profile has an `allreduce` section with a `tp_size` field inside it, but `KernelProfile.AllReduce` is typed as `Lookup1D`. Let me reconcile: the `tp_size` validation can be a separate top-level field, or we embed it. Simplest: add `AllReduceTPSize int` as a top-level field in `KernelProfile`, and keep `AllReduce` as `Lookup1D`.

**Corrected `kernel_profile.go`:**

```go
package latency

import (
    "fmt"
    "math"
    "os"
    "sort"
    "strings"

    "gopkg.in/yaml.v3"
)

// KernelProfile holds pre-computed per-layer latency lookup tables.
// See type comments for the per-field convention.
type KernelProfile struct {
    GPU            string `yaml:"gpu"`
    Backend        string `yaml:"backend"`
    Version        string `yaml:"version"`
    Model          string `yaml:"model"`
    TP             int    `yaml:"tp"`             // TP degree these measurements were taken at
    NumLayers      int    `yaml:"num_layers"`
    NumMoELayers   int    `yaml:"num_moe_layers"`
    NumDenseLayers int    `yaml:"num_dense_layers"`
    HiddenDim      int    `yaml:"hidden_dim"`

    // Tables (per-layer µs, see package-level convention comments above)
    ContextGemm         Lookup1D  `yaml:"context_gemm"`
    ContextAttention    Lookup2D  `yaml:"context_attention"`
    GenerationGemm      Lookup1D  `yaml:"generation_gemm"`
    GenerationAttention Lookup2D  `yaml:"generation_attention"`
    AllReduce           Lookup1D  `yaml:"allreduce"` // token-count axis; 0.0 when TP=1
    MoECompute          *Lookup1D `yaml:"moe_compute,omitempty"` // nil for dense models
}

// Lookup1D maps a dimension (token count) to per-layer latency (µs).
type Lookup1D struct {
    Tokens    []float64 `yaml:"tokens"`
    LatencyUs []float64 `yaml:"latency_us"`
}

// Lookup2D maps two dimensions to per-layer latency (µs).
// LatencyUs[secondary_idx][primary_idx] = latency.
type Lookup2D struct {
    // Primary axis: batch_size (context_attention) or tokens (generation_attention)
    BatchSize []float64 `yaml:"batch_size,omitempty"`
    Tokens    []float64 `yaml:"tokens,omitempty"`
    // Secondary axis: ISL (context_attention) or context length (generation_attention)
    ISL     []float64 `yaml:"isl,omitempty"`
    Context []float64 `yaml:"context,omitempty"`
    // LatencyUs[secondary_idx][primary_idx]
    LatencyUs [][]float64 `yaml:"latency_us"`
}

// LoadKernelProfile reads and validates a kernel_profile.yaml file.
func LoadKernelProfile(path string) (*KernelProfile, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("load kernel profile %q: %w", path, err)
    }
    var p KernelProfile
    if err := yaml.Unmarshal(data, &p); err != nil {
        return nil, fmt.Errorf("parse kernel profile %q: %w", path, err)
    }
    if err := p.validate(); err != nil {
        return nil, fmt.Errorf("kernel profile %q: %w", path, err)
    }
    return &p, nil
}

func (p *KernelProfile) validate() error {
    if p.TP <= 0 {
        return fmt.Errorf("tp must be > 0, got %d", p.TP)
    }
    if p.NumLayers <= 0 {
        return fmt.Errorf("num_layers must be > 0, got %d", p.NumLayers)
    }
    if err := validate1D("context_gemm", p.ContextGemm); err != nil {
        return err
    }
    if err := validate1D("generation_gemm", p.GenerationGemm); err != nil {
        return err
    }
    if err := validate1D("allreduce", p.AllReduce); err != nil {
        return err
    }
    if err := validate2D("context_attention", p.ContextAttention); err != nil {
        return err
    }
    if err := validate2D("generation_attention", p.GenerationAttention); err != nil {
        return err
    }
    if p.MoECompute != nil {
        if err := validate1D("moe_compute", *p.MoECompute); err != nil {
            return err
        }
    }
    return nil
}

func validate1D(name string, t Lookup1D) error {
    if len(t.Tokens) == 0 {
        return fmt.Errorf("%s: tokens must be non-empty", name)
    }
    if len(t.Tokens) != len(t.LatencyUs) {
        return fmt.Errorf("%s: tokens length %d != latency_us length %d", name, len(t.Tokens), len(t.LatencyUs))
    }
    if !sort.Float64sAreSorted(t.Tokens) {
        return fmt.Errorf("%s: tokens must be sorted ascending", name)
    }
    return nil
}

func validate2D(name string, t Lookup2D) error {
    primary := t.primaryAxis()
    secondary := t.secondaryAxis()
    if len(primary) == 0 {
        return fmt.Errorf("%s: primary axis (batch_size or tokens) must be non-empty", name)
    }
    if len(secondary) == 0 {
        return fmt.Errorf("%s: secondary axis (isl or context) must be non-empty", name)
    }
    if len(t.LatencyUs) != len(secondary) {
        return fmt.Errorf("%s: secondary axis length %d != latency_us rows %d", name, len(secondary), len(t.LatencyUs))
    }
    for i, row := range t.LatencyUs {
        if len(row) != len(primary) {
            return fmt.Errorf("%s: row %d length %d != primary axis length %d", name, i, len(row), len(primary))
        }
    }
    return nil
}

func (t *Lookup2D) primaryAxis() []float64 {
    if len(t.BatchSize) > 0 {
        return t.BatchSize
    }
    return t.Tokens
}

func (t *Lookup2D) secondaryAxis() []float64 {
    if len(t.ISL) > 0 {
        return t.ISL
    }
    return t.Context
}

// Interp1D performs linear interpolation. Clamps at boundaries.
func (t Lookup1D) Interp1D(x float64) float64 {
    if len(t.Tokens) == 0 {
        return 0
    }
    xs := t.Tokens
    ys := t.LatencyUs
    if x <= xs[0] {
        return ys[0]
    }
    if x >= xs[len(xs)-1] {
        return ys[len(xs)-1]
    }
    i := sort.SearchFloat64s(xs, x)
    // i is the first index where xs[i] >= x, guaranteed 1 <= i <= len-1
    frac := (x - xs[i-1]) / (xs[i] - xs[i-1])
    return ys[i-1] + frac*(ys[i]-ys[i-1])
}

// Interp2D performs bilinear interpolation.
// primary: batch_size (context) or tokens (generation)
// secondary: isl (context) or context_length (generation)
func (t Lookup2D) Interp2D(primary, secondary float64) float64 {
    pAxis := t.primaryAxis()
    sAxis := t.secondaryAxis()
    if len(pAxis) == 0 || len(sAxis) == 0 {
        return 0
    }

    // interpolate along secondary axis for a given primary index
    interpAtPrimaryIdx := func(pi int) float64 {
        if secondary <= sAxis[0] {
            return t.LatencyUs[0][pi]
        }
        if secondary >= sAxis[len(sAxis)-1] {
            return t.LatencyUs[len(sAxis)-1][pi]
        }
        si := sort.SearchFloat64s(sAxis, secondary)
        frac := (secondary - sAxis[si-1]) / (sAxis[si] - sAxis[si-1])
        return t.LatencyUs[si-1][pi] + frac*(t.LatencyUs[si][pi]-t.LatencyUs[si-1][pi])
    }

    if primary <= pAxis[0] {
        return interpAtPrimaryIdx(0)
    }
    if primary >= pAxis[len(pAxis)-1] {
        return interpAtPrimaryIdx(len(pAxis) - 1)
    }
    pi := sort.SearchFloat64s(pAxis, primary)
    lo := interpAtPrimaryIdx(pi - 1)
    hi := interpAtPrimaryIdx(pi)
    frac := (primary - pAxis[pi-1]) / (pAxis[pi] - pAxis[pi-1])
    return lo + frac*(hi-lo)
}

// clampPositive returns max(0, v) to prevent negative interpolation artifacts.
func clampPositive(v float64) float64 { return math.Max(0, v) }
```

**Step 4: Add interpolation tests**

Add to `kernel_profile_test.go`:

```go
func TestInterp1D_LinearInterpolation(t *testing.T) {
    tbl := Lookup1D{Tokens: []float64{0, 100}, LatencyUs: []float64{0, 200}}
    assert.InDelta(t, 100.0, tbl.Interp1D(50), 0.01)
    assert.InDelta(t, 0.0, tbl.Interp1D(0), 0.01)
    assert.InDelta(t, 200.0, tbl.Interp1D(100), 0.01)
}

func TestInterp1D_Clamps(t *testing.T) {
    tbl := Lookup1D{Tokens: []float64{10, 100}, LatencyUs: []float64{50, 500}}
    assert.InDelta(t, 50.0, tbl.Interp1D(0), 0.01)    // below range
    assert.InDelta(t, 500.0, tbl.Interp1D(1000), 0.01) // above range
}

func TestInterp2D_Bilinear(t *testing.T) {
    tbl := Lookup2D{
        BatchSize: []float64{0, 100},
        ISL:       []float64{0, 100},
        LatencyUs: [][]float64{
            {0, 100},   // ISL=0
            {100, 200}, // ISL=100
        },
    }
    assert.InDelta(t, 100.0, tbl.Interp2D(50, 50), 0.01) // center
    assert.InDelta(t, 0.0, tbl.Interp2D(0, 0), 0.01)
    assert.InDelta(t, 200.0, tbl.Interp2D(100, 100), 0.01)
}
```

**Step 5: Run all tests**

Run: `go test ./sim/latency/... -run "TestLoadKernelProfile|TestInterp" -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add sim/latency/kernel_profile.go sim/latency/kernel_profile_test.go
git commit -m "feat(latency): add kernel profile YAML loader with 1D/2D lookup table interpolation"
```

---

### Task 3: Implement KernelLookupModel

**Files:**
- Create: `sim/latency/kernel_lookup.go`
- Create: `sim/latency/kernel_lookup_test.go`
- Modify: `sim/latency/latency.go` (complete the factory case)

**Architecture decision on factory:**
`NewKernelLookupModel(coeffs, hw)` (called by factory) loads `hw.KernelProfilePath` if set. This requires the caller (CLI) to construct the `ModelHardwareConfig` with `KernelProfilePath` populated using `.WithKernelProfilePath(path)`. If `KernelProfilePath` is empty, returns an error.

**Step 1: Write the failing tests**

Create `sim/latency/kernel_lookup_test.go`:

```go
package latency

import (
    "os"
    "path/filepath"
    "testing"

    sim "github.com/inference-sim/inference-sim/sim"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func writeKernelProfile(t *testing.T) string {
    t.Helper()
    content := `
gpu: h100_sxm
backend: vllm
version: "0.14.0"
model: test-model
tp: 1
num_layers: 32
num_moe_layers: 0
num_dense_layers: 32
hidden_dim: 4096

context_gemm:
  tokens: [1.0, 16.0, 256.0, 4096.0]
  latency_us: [1.5, 1.6, 4.0, 50.0]

context_attention:
  batch_size: [1.0, 4.0, 16.0]
  isl: [128.0, 1024.0, 4096.0]
  latency_us:
    - [0.3, 0.35, 0.4]
    - [0.5, 0.6, 0.75]
    - [1.0, 1.3, 1.8]

generation_gemm:
  tokens: [1.0, 16.0, 256.0]
  latency_us: [1.4, 1.5, 3.8]

generation_attention:
  tokens: [1.0, 16.0, 256.0]
  context: [128.0, 1024.0, 4096.0]
  latency_us:
    - [0.24, 0.28, 0.36]
    - [0.4, 0.5, 0.65]
    - [0.8, 1.1, 1.5]

allreduce:
  tokens: [1.0, 256.0, 4096.0]
  latency_us: [0.0, 0.0, 0.0]
`
    dir := t.TempDir()
    path := filepath.Join(dir, "profile.yaml")
    require.NoError(t, os.WriteFile(path, []byte(content), 0644))
    return path
}

func testKernelLookupModel(t *testing.T) (*KernelLookupModel, error) {
    t.Helper()
    profilePath := writeKernelProfile(t)
    gamma := [10]float64{1, 1, 1, 1, 0, 1, 0, 0, 0, 0} // γ₅=0 (weight), γ₇=0 (moe)
    alpha := [3]float64{500, 0, 0}
    hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "h100", 1, "kernel-lookup", 0).
        WithKernelProfilePath(profilePath)
    coeffs := sim.NewLatencyCoeffs(gamma[:], alpha[:])
    m, err := NewKernelLookupModel(coeffs, hw)
    if err != nil {
        return nil, err
    }
    return m.(*KernelLookupModel), nil
}

func TestKernelLookupModel_EmptyBatch_ReturnsOne(t *testing.T) {
    model, err := testKernelLookupModel(t)
    require.NoError(t, err)
    assert.Equal(t, int64(1), model.StepTime([]*sim.Request{}))
}

func TestKernelLookupModel_QueueingTime(t *testing.T) {
    model, err := testKernelLookupModel(t)
    require.NoError(t, err)
    req := &sim.Request{}
    assert.Equal(t, int64(500), model.QueueingTime(req))
}

func TestKernelLookupModel_SinglePrefillRequest(t *testing.T) {
    model, err := testKernelLookupModel(t)
    require.NoError(t, err)

    req := &sim.Request{
        InputTokens:   make([]int32, 128),
        ProgressIndex: 0,
        NumNewTokens:  128,
    }
    stepTime := model.StepTime([]*sim.Request{req})
    assert.Greater(t, stepTime, int64(0))
    // With 1 prefill request, 128 tokens, ISL=128, numLayers=32:
    // T_pf_gemm = Interp1D(128) * 32 ≈ some positive value
    // T_pf_attn = Interp2D(batchSize=1, avgISL=128) * 32
    // Both positive, so stepTime > 0 guaranteed.
}

func TestKernelLookupModel_FactoryError_NoProfile(t *testing.T) {
    hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0)
    coeffs := sim.NewLatencyCoeffs(make([]float64, 10), make([]float64, 3))
    _, err := NewKernelLookupModel(coeffs, hw)
    assert.Error(t, err, "should fail without KernelProfilePath")
}

func TestKernelLookupModel_FactoryError_TooFewCoeffs(t *testing.T) {
    profilePath := writeKernelProfile(t)
    hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0).
        WithKernelProfilePath(profilePath)
    // Only 5 beta coefficients instead of 10
    coeffs := sim.NewLatencyCoeffs(make([]float64, 5), make([]float64, 3))
    _, err := NewKernelLookupModel(coeffs, hw)
    assert.Error(t, err, "should fail with fewer than 10 gamma coefficients")
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/latency/... -run TestKernelLookupModel -v`
Expected: FAIL with compile error (NewKernelLookupModel undefined)

**Step 3: Implement `kernel_lookup.go`**

Create `sim/latency/kernel_lookup.go`:

```go
package latency

import (
    "fmt"

    "github.com/inference-sim/inference-sim/sim"
    "github.com/inference-sim/inference-sim/sim/internal/util"
)

// KernelLookupModel predicts step time using per-layer latency lookup tables
// from aiconfigurator's measured GPU kernel database, corrected by learned γ factors.
//
// Step-time formula:
//
//     γ₁·T_pf_gemm + γ₂·T_pf_attn + γ₃·T_dc_gemm + γ₄·T_dc_attn
//     + γ₆·T_allreduce + γ₇·T_moe
//     + γ₈·numLayers + γ₉·batchSize + γ₁₀
//
// γ₁-γ₄ and γ₆-γ₇: dimensionless corrections (expected ~1.0)
// γ₈: µs/layer overhead; γ₉: µs/request overhead; γ₁₀: µs/step overhead
//
// Note: γ₅ (weight loading) is not used — weight memory access is already
// captured in the GEMM measurements, so a separate T_weight term would double-count.
//
// Basis function conventions (enforced by profile validation):
//   - T_pf_gemm: context GEMM, interpolated by totalPrefillTokens, multiplied by numLayers
//   - T_pf_attn: context attention, interpolated by (numPrefillRequests, avgISL), × numLayers
//   - T_dc_gemm: generation GEMM, interpolated by totalDecodeTokens, × numLayers
//   - T_dc_attn: generation attention, interpolated by (totalDecodeTokens, avgDecodeCtx), × numLayers
//   - T_allreduce: per-step communication, interpolated by totalTokens, × allReduceUnits
//     where allReduceUnits = 2·numDenseLayers + 1·numMoELayers
//   - T_moe: expert computation, interpolated by totalTokens, × numMoELayers
type KernelLookupModel struct {
    gamma [10]float64 // γ₁-γ₁₀ (γ₅ unused but stored for coefficient index alignment)
    alpha [3]float64  // α₀ (queueing), α₁ (post-decode), α₂ (per-output-token)

    // Pre-loaded lookup tables (per-layer µs)
    contextGemm         Lookup1D
    contextAttn         Lookup2D
    generationGemm      Lookup1D
    generationAttn      Lookup2D
    allreduce           Lookup1D
    moeCompute          *Lookup1D

    // Architecture (from kernel profile)
    numLayers       int
    numMoELayers    int
    numDenseLayers  int
    allReduceUnits  int // 2·numDenseLayers + 1·numMoELayers
}

// NewKernelLookupModel creates a KernelLookupModel from BLIS config types.
// Implements the NewLatencyModel factory interface.
//
// Requires:
//   - hw.KernelProfilePath != ""  (set via hw.WithKernelProfilePath(path))
//   - len(coeffs.BetaCoeffs) >= 10  (γ₁-γ₁₀)
//   - len(coeffs.AlphaCoeffs) >= 3  (α₀-α₂)
func NewKernelLookupModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
    if hw.KernelProfilePath == "" {
        return nil, fmt.Errorf("kernel-lookup: KernelProfilePath must be set; " +
            "use hw.WithKernelProfilePath(path) or --kernel-profile flag")
    }
    if len(coeffs.BetaCoeffs) < 10 {
        return nil, fmt.Errorf("kernel-lookup: requires 10 gamma coefficients " +
            "(β₁-β₁₀ slot for γ₁-γ₁₀), got %d", len(coeffs.BetaCoeffs))
    }
    if len(coeffs.AlphaCoeffs) < 3 {
        return nil, fmt.Errorf("kernel-lookup: requires 3 alpha coefficients, got %d",
            len(coeffs.AlphaCoeffs))
    }
    if err := validateCoeffs("BetaCoeffs/gamma", coeffs.BetaCoeffs[:10]); err != nil {
        return nil, err
    }

    profile, err := LoadKernelProfile(hw.KernelProfilePath)
    if err != nil {
        return nil, fmt.Errorf("kernel-lookup: %w", err)
    }

    // Validate that runtime TP matches profile TP.
    // If hw.TP == 0 (e.g., blackbox-mode callers), skip validation.
    if hw.TP > 0 && hw.TP != profile.TP {
        return nil, fmt.Errorf("kernel-lookup: runtime TP=%d does not match profile TP=%d (profile: %s)",
            hw.TP, profile.TP, hw.KernelProfilePath)
    }

    var gamma [10]float64
    copy(gamma[:], coeffs.BetaCoeffs[:10])
    var alpha [3]float64
    copy(alpha[:], coeffs.AlphaCoeffs[:3])

    numDense := profile.NumDenseLayers
    numMoE := profile.NumMoELayers
    if numDense == 0 && numMoE == 0 {
        // Fallback: treat all layers as dense
        numDense = profile.NumLayers
    }

    return &KernelLookupModel{
        gamma:          gamma,
        alpha:          alpha,
        contextGemm:    profile.ContextGemm,
        contextAttn:    profile.ContextAttention,
        generationGemm: profile.GenerationGemm,
        generationAttn: profile.GenerationAttention,
        allreduce:      profile.AllReduce,
        moeCompute:     profile.MoECompute,
        numLayers:      profile.NumLayers,
        numMoELayers:   numMoE,
        numDenseLayers: numDense,
        allReduceUnits: 2*numDense + numMoE,
    }, nil
}

// StepTime computes step time via interpolated kernel lookups.
// Single O(batch_size) pass, zero heap allocations.
func (m *KernelLookupModel) StepTime(batch []*sim.Request) int64 {
    if len(batch) == 0 {
        return 1
    }

    var (
        totalPrefillTokens float64
        numPrefillRequests float64
        sumPrefillISL      float64
        totalDecodeTokens  float64
        sumDecodeCtx       float64
    )
    batchSize := float64(len(batch))
    L := float64(m.numLayers)

    for _, req := range batch {
        if req.ProgressIndex < util.Len64(req.InputTokens) {
            totalPrefillTokens += float64(req.NumNewTokens)
            numPrefillRequests++
            sumPrefillISL += float64(len(req.InputTokens))
        } else if len(req.OutputTokens) > 0 {
            totalDecodeTokens++
            sumDecodeCtx += float64(req.ProgressIndex)
        }
    }

    avgPrefillISL := float64(0)
    if numPrefillRequests > 0 {
        avgPrefillISL = sumPrefillISL / numPrefillRequests
    }
    avgDecodeCtx := float64(0)
    if totalDecodeTokens > 0 {
        avgDecodeCtx = sumDecodeCtx / totalDecodeTokens
    }
    totalTokens := totalPrefillTokens + totalDecodeTokens

    // γ₁·T_pf_gemm: context GEMM × numLayers
    var tPfGemm float64
    if totalPrefillTokens > 0 {
        tPfGemm = clampPositive(m.contextGemm.Interp1D(totalPrefillTokens)) * L
    }

    // γ₂·T_pf_attn: context attention × numLayers
    // Primary axis: numPrefillRequests (= batch_size in aiconfigurator's context_attention)
    // Secondary axis: avgPrefillISL
    var tPfAttn float64
    if numPrefillRequests > 0 {
        tPfAttn = clampPositive(m.contextAttn.Interp2D(numPrefillRequests, avgPrefillISL)) * L
    }

    // γ₃·T_dc_gemm: generation GEMM × numLayers
    var tDcGemm float64
    if totalDecodeTokens > 0 {
        tDcGemm = clampPositive(m.generationGemm.Interp1D(totalDecodeTokens)) * L
    }

    // γ₄·T_dc_attn: generation attention × numLayers
    // Primary axis: totalDecodeTokens (= decode batch size, 1 token/request)
    // Secondary axis: avgDecodeCtx
    var tDcAttn float64
    if totalDecodeTokens > 0 {
        tDcAttn = clampPositive(m.generationAttn.Interp2D(totalDecodeTokens, avgDecodeCtx)) * L
    }

    // γ₆·T_allreduce: TP AllReduce × allReduceUnits (2·dense + 1·MoE)
    // allReduceUnits = 0 when TP=1 or no AllReduce data
    var tAllReduce float64
    if m.allReduceUnits > 0 && totalTokens > 0 {
        tAllReduce = clampPositive(m.allreduce.Interp1D(totalTokens)) * float64(m.allReduceUnits)
    }

    // γ₇·T_moe: MoE expert computation × numMoELayers
    var tMoE float64
    if m.moeCompute != nil && m.numMoELayers > 0 && totalTokens > 0 {
        tMoE = clampPositive(m.moeCompute.Interp1D(totalTokens)) * float64(m.numMoELayers)
    }

    stepTime := m.gamma[0]*tPfGemm +
        m.gamma[1]*tPfAttn +
        m.gamma[2]*tDcGemm +
        m.gamma[3]*tDcAttn +
        // gamma[4] = γ₅ (weight loading) — intentionally unused; set to 0 in training
        m.gamma[5]*tAllReduce +
        m.gamma[6]*tMoE +
        m.gamma[7]*L +
        m.gamma[8]*batchSize +
        m.gamma[9]

    return max(1, clampToInt64(stepTime))
}

func (m *KernelLookupModel) QueueingTime(req *sim.Request) int64 {
    return clampToInt64(m.alpha[0])
}

func (m *KernelLookupModel) OutputTokenProcessingTime() int64 {
    return clampToInt64(m.alpha[2])
}

func (m *KernelLookupModel) PostDecodeFixedOverhead() int64 {
    return clampToInt64(m.alpha[1])
}
```

**Step 4: Run tests**

Run: `go test ./sim/latency/... -run TestKernelLookupModel -v`
Expected: All PASS

Run: `go build ./...`
Expected: PASS (factory case now compiles)

**Step 5: Commit**

```bash
git add sim/latency/kernel_lookup.go sim/latency/kernel_lookup_test.go
git commit -m "feat(latency): implement KernelLookupModel with per-layer interpolated kernel lookups"
```

---

### Task 4: Wire CLI Flags for Kernel-Lookup Backend

**Files:**
- Modify: `cmd/root.go` (add `--kernel-profile` flag; update `resolveLatencyConfig`)
- Modify: `cmd/simconfig_shared_test.go` (add flag parity)

**Step 1: Add package-level var and flag registration**

In `cmd/root.go`, add to the var block:

```go
var kernelProfilePath string
```

In `registerSimConfigFlags`, add:

```go
cmd.Flags().StringVar(&kernelProfilePath, "kernel-profile", "",
    "path to kernel_profile.yaml for --latency-model kernel-lookup")
```

**Step 2: Add kernel-lookup resolution to `resolveLatencyConfig`**

After the `latency-model` validation block (where `backend` is determined),
add handling for kernel-lookup. The key change: for kernel-lookup, the
`ModelHardwareConfig` must have `KernelProfilePath` set. The resolution returns
a `latencyResolution` with `Backend="kernel-lookup"` and `KernelProfilePath` set.

Extend `latencyResolution` struct:

```go
type latencyResolution struct {
    Backend           string
    ModelConfig       sim.ModelConfig
    HWConfig          sim.HardwareCalib
    AlphaCoeffs       []float64
    BetaCoeffs        []float64
    KernelProfilePath string // non-empty only for kernel-lookup
}
```

In `resolveLatencyConfig`, after loading alpha/beta:

```go
if backend == "kernel-lookup" {
    if !cmd.Flags().Changed("kernel-profile") || kernelProfilePath == "" {
        logrus.Fatalf("--kernel-profile is required for --latency-model kernel-lookup")
    }
    if len(beta) < 10 {
        logrus.Fatalf("kernel-lookup requires 10 gamma coefficients (--beta-coeffs), got %d", len(beta))
    }
    // ... existing model config loading for numLayers etc. (same as trained-roofline path)
    // The actual profile loading happens in NewKernelLookupModel at factory call time.
    return latencyResolution{
        Backend:           "kernel-lookup",
        AlphaCoeffs:       alpha,
        BetaCoeffs:        beta,
        KernelProfilePath: kernelProfilePath,
    }
}
```

**Step 3: Update the ModelHardwareConfig construction site**

Where `resolveLatencyConfig` result feeds into `NewModelHardwareConfig` (in `runCmd`
and `replayCmd` run functions), apply the profile path:

```go
hw := sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tp, lr.Backend, maxModelLen)
if lr.KernelProfilePath != "" {
    hw = hw.WithKernelProfilePath(lr.KernelProfilePath)
}
```

**Step 4: Add test for flag parity**

In `cmd/simconfig_shared_test.go`, add `"kernel-profile"` to `sharedFlags` in
`TestBothCommands_SimConfigFlagsHaveIdenticalDefaults`.

**Step 5: Run tests**

Run: `go test ./cmd/... -v -run TestBothCommands_SimConfigFlagsHaveIdenticalDefaults`
Expected: PASS

Run: `go build ./...`
Expected: PASS

**Step 6: Commit**

```bash
git add cmd/root.go cmd/simconfig_shared_test.go
git commit -m "feat(cli): add --kernel-profile flag and kernel-lookup resolution path"
```

---

### Task 5: Python Script — Generate Kernel Profiles

**Files:**
- Create: `training/scripts/generate_kernel_profile.py`

**Prerequisites:** The aiconfigurator CSV data files must be available via Git LFS. Run `git lfs pull` in the repo directory before running this script. The generated YAML files should be committed to the repo so Go tests can run without LFS.

**Step 1: Implement the script**

Create `training/scripts/generate_kernel_profile.py`:

```python
#!/usr/bin/env python3
"""Generate kernel_profile.yaml for the BLIS kernel-lookup latency backend.

Queries aiconfigurator's measured GPU operations database to produce per-layer
latency lookup tables for a given model/GPU/TP configuration.

PREREQUISITES: aiconfigurator data files must be checked out via Git LFS.
  cd /path/to/repo && git lfs pull

Usage:
    cd inference-sim  # repo root
    python training/scripts/generate_kernel_profile.py \
        --model meta-llama/Llama-2-7b-hf \
        --gpu h100_sxm \
        --backend vllm \
        --version 0.14.0 \
        --tp 1 \
        --output training/kernel_profiles/llama-2-7b-tp1.yaml

    # Batch generate for all training experiments:
    python training/scripts/generate_kernel_profile.py --from-exp-dir training/trainval_data/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Token count grid for GEMM and MoE lookups
TOKEN_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Batch size grid for context attention
BATCH_GRID = [1, 2, 4, 8, 16, 32, 64, 128]

# ISL grid for context attention secondary axis
ISL_GRID = [64, 128, 256, 512, 1024, 2048, 4096]

# Context length grid for generation attention secondary axis
CTX_GRID = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def add_aiconfigurator_to_path():
    """Add the aiconfigurator package to sys.path."""
    repo_root = Path(__file__).parent.parent.parent
    src_path = repo_root / "aiconfigurator" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


add_aiconfigurator_to_path()

from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.common import GEMMQuantMode, CommQuantMode, FMHAQuantMode, KVCacheQuantMode
from aiconfigurator.sdk import utils as sdk_utils


def load_model_info(model_path: str, hf_cache_dir: str = None):
    """Load HF config.json and extract architectural parameters."""
    info = sdk_utils._get_model_info(model_path)
    config = info["config"]
    return {
        "num_layers": config["layers"],
        "hidden_size": config["hidden_size"],
        "num_heads": config["n"],
        "num_kv_heads": config.get("n_kv", config["n"]),
        "head_dim": config.get("d", config["hidden_size"] // config["n"]),
        "inter_size": config["inter_size"],
        "vocab_size": config["vocab_size"],
        "num_experts": config.get("num_experts", 0),
        "topk": config.get("topk", 0),
        "moe_inter_size": config.get("moe_inter_size", config["inter_size"]),
        "architecture": config.get("architecture", ""),
    }


def compute_gemm_dimensions(model_info: dict, tp: int):
    """Compute per-layer GEMM shapes (n, k) for each op type."""
    h = model_info["hidden_size"]
    n_heads = model_info["num_heads"]
    n_kv_heads = model_info["num_kv_heads"]
    head_dim = model_info["head_dim"]
    inter = model_info["inter_size"]
    num_experts = model_info["num_experts"]
    topk = model_info["topk"]
    moe_inter = model_info["moe_inter_size"]

    kv_heads_per_gpu = max(1, n_kv_heads // tp)
    heads_per_gpu = n_heads // tp

    gemms = {
        # QKV projection: fused Q + K + V
        "qkv": {
            "n": heads_per_gpu * head_dim + 2 * kv_heads_per_gpu * head_dim,
            "k": h,
        },
        # Attention output projection
        "proj": {
            "n": h,
            "k": heads_per_gpu * head_dim,
        },
        # FFN gate + up (SwiGLU fused)
        "gate_up": {
            "n": 2 * inter // tp,
            "k": h,
        },
        # FFN down
        "down": {
            "n": h,
            "k": inter // tp,
        },
    }
    return gemms


def query_context_gemm(db: PerfDatabase, gemms: dict, token_grid: list, quant_mode) -> list:
    """Query total per-layer context GEMM latency summed over all GEMM types."""
    latencies = []
    for tokens in token_grid:
        total = 0.0
        for name, dims in gemms.items():
            result = db.query_gemm(m=tokens, n=dims["n"], k=dims["k"], quant_mode=quant_mode)
            total += float(result)  # ms → will convert to µs below
        latencies.append(total * 1000)  # ms → µs
    return latencies


def query_generation_gemm(db: PerfDatabase, gemms: dict, token_grid: list, quant_mode) -> list:
    """Query total per-layer generation GEMM latency (same shapes, different m values)."""
    # Generation uses small m (batch size), same n/k as context
    return query_context_gemm(db, gemms, token_grid, quant_mode)


def query_context_attention(db: PerfDatabase, model_info: dict, tp: int,
                            batch_grid: list, isl_grid: list,
                            fmha_quant_mode, kv_quant_mode) -> list:
    """Query 2D context attention table. Returns list of rows (one per ISL value)."""
    heads_per_gpu = model_info["num_heads"] // tp
    kv_per_gpu = max(1, model_info["num_kv_heads"] // tp)
    head_dim = model_info["head_dim"]

    rows = []
    for isl in isl_grid:
        row = []
        for b in batch_grid:
            result = db.query_context_attention(
                quant_mode=fmha_quant_mode,
                kv_cache_dtype=kv_quant_mode,
                b=b, s=isl,
                n=heads_per_gpu, kv_n=kv_per_gpu,
                head_size=head_dim, window_size=0,
            )
            row.append(float(result) * 1000)  # ms → µs
        rows.append(row)
    return rows


def query_generation_attention(db: PerfDatabase, model_info: dict, tp: int,
                                token_grid: list, ctx_grid: list,
                                kv_quant_mode) -> list:
    """Query 2D generation attention table. Returns list of rows (one per context value)."""
    heads_per_gpu = model_info["num_heads"] // tp
    kv_per_gpu = max(1, model_info["num_kv_heads"] // tp)
    head_dim = model_info["head_dim"]

    rows = []
    for ctx in ctx_grid:
        row = []
        for tokens in token_grid:
            # tokens = decode batch size (1 per request)
            result = db.query_generation_attention(
                kv_cache_dtype=kv_quant_mode,
                b=int(tokens), s=ctx,
                n=heads_per_gpu, kv_n=kv_per_gpu,
                head_size=head_dim, window_size=0,
            )
            row.append(float(result) * 1000)
        rows.append(row)
    return rows


def query_allreduce(db: PerfDatabase, model_info: dict, tp: int, token_grid: list) -> list:
    """Query AllReduce latency keyed by token count (converted from message_size).

    AllReduce message size = tokens × hidden_size (element count).
    We query at these message sizes and store the result indexed by token count
    for the Go runtime.

    Returns 0.0 for all entries when TP=1 (no AllReduce needed).
    """
    if tp == 1:
        return [0.0] * len(token_grid)

    h = model_info["hidden_size"]
    latencies = []
    for tokens in token_grid:
        msg_size = int(tokens * h)
        result = db.query_custom_allreduce(
            dtype=CommQuantMode.half, tp_size=tp, size=msg_size
        )
        latencies.append(float(result) * 1000)  # ms → µs
    return latencies


def generate_kernel_profile(
    model_path: str,
    gpu: str,
    backend: str,
    version: str,
    tp: int,
    systems_root: str,
    output_path: str,
):
    """Generate kernel_profile.yaml for the given model/GPU/TP configuration."""
    print(f"Loading model info: {model_path}")
    model_info = load_model_info(model_path)
    num_layers = model_info["num_layers"]
    print(f"  layers={num_layers}, hidden={model_info['hidden_size']}, "
          f"heads={model_info['num_heads']}, experts={model_info['num_experts']}")

    print(f"Loading aiconfigurator PerfDatabase: {gpu}/{backend}/{version}")
    db = PerfDatabase(
        system=gpu,
        backend=backend,
        version=version,
        systems_root=systems_root,
    )

    # Determine quantization mode from model config
    # FP8 for FP8 models (like Scout FP8), FP16 otherwise
    is_fp8 = "fp8" in model_path.lower() or "fp8" in model_info.get("architecture", "").lower()
    quant_mode = GEMMQuantMode.fp8 if is_fp8 else GEMMQuantMode.float16
    fmha_mode = FMHAQuantMode.fp8 if is_fp8 else FMHAQuantMode.float16
    kv_mode = KVCacheQuantMode.fp8 if is_fp8 else KVCacheQuantMode.fp16

    gemm_dims = compute_gemm_dimensions(model_info, tp)

    print("Querying context GEMM...")
    ctx_gemm_lat = query_context_gemm(db, gemm_dims, TOKEN_GRID, quant_mode)
    ctx_gemm_per_layer = [v / num_layers for v in ctx_gemm_lat]

    print("Querying context attention...")
    ctx_attn_rows = query_context_attention(db, model_info, tp,
                                            BATCH_GRID, ISL_GRID, fmha_mode, kv_mode)
    ctx_attn_per_layer = [[v / num_layers for v in row] for row in ctx_attn_rows]

    print("Querying generation GEMM...")
    gen_gemm_lat = query_generation_gemm(db, gemm_dims,
                                          [t for t in TOKEN_GRID if t <= 512], quant_mode)
    gen_token_grid = [t for t in TOKEN_GRID if t <= 512]
    gen_gemm_per_layer = [v / num_layers for v in gen_gemm_lat]

    print("Querying generation attention...")
    gen_attn_rows = query_generation_attention(db, model_info, tp,
                                               gen_token_grid, CTX_GRID, kv_mode)
    gen_attn_per_layer = [[v / num_layers for v in row] for row in gen_attn_rows]

    print("Querying AllReduce...")
    allreduce_lat = query_allreduce(db, model_info, tp, TOKEN_GRID)
    allreduce_per_layer = [v / num_layers for v in allreduce_lat]

    # MoE: only for models with experts
    moe_compute = None
    num_moe_layers = 0
    num_dense_layers = num_layers
    if model_info["num_experts"] > 0:
        num_experts = model_info["num_experts"]
        topk = model_info["topk"]
        moe_inter = model_info["moe_inter_size"]
        h = model_info["hidden_size"]
        # For interleaved MoE (Scout): num_moe_layers = num_layers // 2
        # For uniform MoE (Mixtral): num_moe_layers = num_layers
        # Heuristic: if "scout" or "16e" in model name, assume interleaved
        is_interleaved = any(x in model_path.lower() for x in ["scout", "16e"])
        if is_interleaved:
            num_moe_layers = num_layers // 2
        else:
            num_moe_layers = num_layers
        num_dense_layers = num_layers - num_moe_layers

        print(f"Querying MoE compute (experts={num_experts}, topk={topk})...")
        moe_lats = []
        for tokens in TOKEN_GRID:
            result = db.query_moe(
                num_tokens=tokens, hidden_size=h, inter_size=moe_inter,
                topk=topk, num_experts=num_experts,
                moe_tp_size=tp, moe_ep_size=1,
                quant_mode=quant_mode if quant_mode != GEMMQuantMode.float16 else None,
                workload_distribution="uniform", is_context=(tokens > 1),
            )
            moe_lats.append(float(result) * 1000)
        moe_per_layer = [v / max(1, num_moe_layers) for v in moe_lats]
        moe_compute = {
            "tokens": [float(t) for t in TOKEN_GRID],
            "latency_us": [round(v, 4) for v in moe_per_layer],
        }

    profile = {
        "gpu": gpu,
        "backend": backend,
        "version": version,
        "model": model_path,
        "tp": tp,
        "num_layers": num_layers,
        "num_moe_layers": num_moe_layers,
        "num_dense_layers": num_dense_layers,
        "hidden_dim": model_info["hidden_size"],
        "context_gemm": {
            "tokens": [float(t) for t in TOKEN_GRID],
            "latency_us": [round(v, 4) for v in ctx_gemm_per_layer],
        },
        "context_attention": {
            "batch_size": [float(b) for b in BATCH_GRID],
            "isl": [float(s) for s in ISL_GRID],
            "latency_us": [[round(v, 4) for v in row] for row in ctx_attn_per_layer],
        },
        "generation_gemm": {
            "tokens": [float(t) for t in gen_token_grid],
            "latency_us": [round(v, 4) for v in gen_gemm_per_layer],
        },
        "generation_attention": {
            "tokens": [float(t) for t in gen_token_grid],
            "context": [float(c) for c in CTX_GRID],
            "latency_us": [[round(v, 4) for v in row] for row in gen_attn_per_layer],
        },
        "allreduce": {
            "tokens": [float(t) for t in TOKEN_GRID],
            "latency_us": [round(v, 4) for v in allreduce_per_layer],
        },
    }
    if moe_compute is not None:
        profile["moe_compute"] = moe_compute

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="h100_sxm")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--version", default="0.14.0")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--systems-root", default=None,
                        help="Path to aiconfigurator/src/aiconfigurator/systems/ "
                             "(auto-detected if not specified)")
    parser.add_argument("--from-exp-dir", default=None,
                        help="Generate profiles for all experiments in this directory")
    args = parser.parse_args()

    # Auto-detect systems root
    if args.systems_root is None:
        script_dir = Path(__file__).parent
        args.systems_root = str(script_dir.parent.parent / "aiconfigurator" /
                                "src" / "aiconfigurator" / "systems")

    if args.from_exp_dir:
        generate_from_exp_dir(args.from_exp_dir, args.gpu, args.backend,
                              args.version, args.systems_root, args.output)
    else:
        generate_kernel_profile(args.model, args.gpu, args.backend,
                                args.version, args.tp, args.systems_root, args.output)


def generate_from_exp_dir(exp_dir: str, gpu: str, backend: str, version: str,
                           systems_root: str, output_dir: str):
    """Generate profiles for all experiments in a training data directory."""
    import glob as glob_module
    for config_path in glob_module.glob(os.path.join(exp_dir, "*/exp-config.yaml")):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model = config["model"]
        tp = config["tensor_parallelism"]
        exp_name = os.path.basename(os.path.dirname(config_path))
        output_path = os.path.join(output_dir, f"{exp_name}.yaml")
        print(f"\n=== {exp_name} ===")
        try:
            generate_kernel_profile(model, gpu, backend, version, tp,
                                    systems_root, output_path)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Run for one experiment**

```bash
cd /Users/sri/Documents/Projects/inference-sim
python training/scripts/generate_kernel_profile.py \
    --model meta-llama/Llama-2-7b-hf \
    --tp 1 \
    --output training/kernel_profiles/20260217-155451-llama-2-7b-tp1-codegen.yaml
```

Expected: Produces a valid YAML file. Verify with:
```bash
head -20 training/kernel_profiles/20260217-155451-llama-2-7b-tp1-codegen.yaml
```

**Step 3: Validate the profile loads in Go**

Run: `go test ./sim/latency/... -run TestLoadKernelProfile -v`
Expected: PASS (the generated file should pass validation)

**Step 4: Generate profiles for all 15 training experiments**

```bash
python training/scripts/generate_kernel_profile.py \
    --from-exp-dir training/trainval_data/ \
    --output training/kernel_profiles/
```

**Step 5: Commit profiles**

```bash
git add training/scripts/generate_kernel_profile.py training/kernel_profiles/
git commit -m "feat(training): generate aiconfigurator kernel profiles for all 15 training experiments"
```

---

### Task 6: Training Loop Integration and iter30 Optimization

**Files:**
- Modify: `training/run_blis_and_compute_loss.py` (add `--kernel-profile` passthrough)
- Create: `training/iterations/iter30/coefficient_bounds.yaml`
- Create: `training/iterations/iter30/iter30-FINDINGS.md` (after running)

**Step 1: Update training runner for kernel-lookup**

In `training/run_blis_and_compute_loss.py`, the runner invokes the BLIS binary with
`--latency-model evolved` (or whatever is configured). Add support for passing
`--latency-model kernel-lookup --kernel-profile <path>` where the profile path
comes from `training/kernel_profiles/<experiment_name>.yaml`.

Map from experiment folder name to kernel profile path:
```python
def get_kernel_profile_path(experiment_folder: str, profiles_dir: str) -> str:
    exp_name = os.path.basename(experiment_folder)
    profile_path = os.path.join(profiles_dir, f"{exp_name}.yaml")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"No kernel profile for {exp_name}: {profile_path}")
    return profile_path
```

Add CLI args:
```
--kernel-profiles-dir: directory containing kernel_profile YAML files
```

When `--latency-model kernel-lookup`, pass `--kernel-profile <path>` to BLIS binary.

**Step 2: Create coefficient bounds**

Create `training/iterations/iter30/coefficient_bounds.yaml`:

```yaml
# iter30 gamma coefficients for kernel-lookup backend
# gamma[0..9] maps to β₁-β₁₀ in the BLIS --beta-coeffs flag
# gamma[0..3]: dimensionless corrections for prefill/decode GEMM/attention
# gamma[4]: UNUSED (weight loading — removed to avoid double-counting)
# gamma[5..6]: allreduce and MoE corrections
# gamma[7..9]: per-layer, per-request, per-step additive overheads (µs)

initial_gamma: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 40.0, 3.0, 100.0]
initial_alpha: [0.0, 0.0, 0.0]

bounds:
  # [min, max] per coefficient
  gamma:
    - [0.5, 2.0]   # γ₁: prefill GEMM correction
    - [0.5, 2.0]   # γ₂: prefill attention correction
    - [0.5, 2.0]   # γ₃: decode GEMM correction
    - [0.5, 2.0]   # γ₄: decode attention correction
    - [0.0, 0.0]   # γ₅: FIXED at 0 (weight loading removed)
    - [0.0, 2.0]   # γ₆: allreduce correction (0 when TP=1)
    - [0.0, 5.0]   # γ₇: MoE correction (0 for dense, free for MoE)
    - [0.0, 200.0] # γ₈: per-layer overhead (µs)
    - [0.0, 50.0]  # γ₉: per-request overhead (µs)
    - [0.0, 500.0] # γ₁₀: per-step overhead (µs)
  alpha:
    - [0.0, 1000.0]  # α₀: queueing time (µs)
    - [0.0, 1000.0]  # α₁: post-decode fixed overhead (µs)
    - [0.0, 100.0]   # α₂: per-output-token processing (µs)
```

**Step 3: Run baseline evaluation with initial γ**

```bash
cd training
python run_blis_and_compute_loss.py \
    --latency-model kernel-lookup \
    --kernel-profiles-dir kernel_profiles/ \
    --beta-coeffs 1.0,1.0,1.0,1.0,0.0,1.0,0.0,40.0,3.0,100.0 \
    --alpha-coeffs 0.0,0.0,0.0 \
    --evaluate-per-experiment \
    --max-workers 15
```

Record the baseline loss in `iter30-FINDINGS.md`.

**Step 4: Optimize γ with golden section search**

Use `iter29_golden_section.py` as a template. Optimize the free parameters:
γ₁, γ₂, γ₃, γ₄, γ₆, γ₇, γ₈, γ₉, γ₁₀ (γ₅ fixed at 0).

**Step 5: Record findings**

Fill in `training/iterations/iter30/iter30-FINDINGS.md` with actual results.

---

### Task 7: End-to-End Validation

**Files:**
- Modify: `sim/latency/latency_test.go` (add kernel-lookup to all-backends test)

**Step 1: Add kernel-lookup to all-backends tests**

Find `TestAllBackends_StepTime_EmptyBatch_FloorAtOne` in `sim/latency/latency_test.go`
and add a kernel-lookup test case using `NewKernelLookupModel` with a valid profile path.

```go
// Use a synthetic profile written to a temp file
profilePath := kernelLookupTestProfilePath(t) // helper that writes testProfileYAML
hw := sim.NewModelHardwareConfig(..., "kernel-lookup", 0).WithKernelProfilePath(profilePath)
coeffs := sim.NewLatencyCoeffs(make([]float64, 10), []float64{0, 0, 0})
m, err := latency.NewLatencyModel(coeffs, hw)
require.NoError(t, err)
assert.Equal(t, int64(1), m.StepTime([]*sim.Request{}))
```

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: All tests PASS

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No issues

**Step 4: Commit**

```bash
git commit -am "test(latency): add kernel-lookup to all-backends validation tests"
```

---

## Execution Order

```
Task 1 (register backend) → Task 2 (lookup tables) → Task 3 (KernelLookupModel) → Task 4 (CLI) → Task 7 (validation)
                                                                                       ↑
Task 5 (Python profiles, needs LFS) ──────────────────────────────────────────────→ Task 6 (training)
```

Tasks 1-4 and 7 are pure Go with synthetic test profiles — they do NOT require LFS data.
Task 5 requires LFS data checkout. Task 6 depends on Task 5.

## Key Invariants

- **Per-layer convention**: ALL lookup tables store per-layer latencies. Multipliers in `StepTime` are the correct layer count for each term.
- **AllReduce units**: `2·numDenseLayers + numMoELayers` (not `numLayers`). TP=1 → all zeros, units don't matter.
- **γ₅ = 0**: Weight loading is not a separate term (captured in GEMM measurements).
- **Context attention batch axis**: Matches aiconfigurator's `b=batch_size` (number of prefill requests), NOT total token count.
- **Profile TP validation**: Runtime must match profile TP — enforced in `NewKernelLookupModel`.
