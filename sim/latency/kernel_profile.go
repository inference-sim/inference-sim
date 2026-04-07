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
//   - context_attention.batch_size × isl: prefill batch_size and avg ISL
//   - generation_gemm.tokens: decode token count (== decode batch size, 1 token/request)
//   - generation_attention.tokens × context: decode batch_size and avg context length
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

// Lookup1D maps a token count to per-layer latency (us).
// Tokens must be non-empty and sorted ascending.
type Lookup1D struct {
	Tokens    []float64 `yaml:"tokens"`
	LatencyUs []float64 `yaml:"latency_us"`
}

// Lookup2D maps two dimensions to per-layer latency (us).
// LatencyUs[secondary_idx][primary_idx] = latency.
// For context_attention: primary=BatchSize, secondary=ISL.
// For generation_attention: primary=Tokens, secondary=Context.
type Lookup2D struct {
	BatchSize []float64   `yaml:"batch_size,omitempty"`
	Tokens    []float64   `yaml:"tokens,omitempty"`
	ISL       []float64   `yaml:"isl,omitempty"`
	Context   []float64   `yaml:"context,omitempty"`
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
		return fmt.Errorf("%s: tokens length %d != latency_us length %d",
			name, len(t.Tokens), len(t.LatencyUs))
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
		return fmt.Errorf("%s: secondary axis length %d != latency_us rows %d",
			name, len(secondary), len(t.LatencyUs))
	}
	for i, row := range t.LatencyUs {
		if len(row) != len(primary) {
			return fmt.Errorf("%s: row %d length %d != primary axis length %d",
				name, i, len(row), len(primary))
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

// Interp1D performs linear interpolation on a 1D lookup table.
// Clamps to boundary values when x is outside the measured range.
func (t Lookup1D) Interp1D(x float64) float64 {
	xs := t.Tokens
	ys := t.LatencyUs
	if len(xs) == 0 {
		return 0
	}
	if x <= xs[0] {
		return ys[0]
	}
	if x >= xs[len(xs)-1] {
		return ys[len(xs)-1]
	}
	i := sort.SearchFloat64s(xs, x)
	// i is the smallest index where xs[i] >= x; guaranteed 1 <= i < len(xs)
	frac := (x - xs[i-1]) / (xs[i] - xs[i-1])
	return ys[i-1] + frac*(ys[i]-ys[i-1])
}

// Interp2D performs bilinear interpolation on a 2D lookup table.
// primary: batch_size (context_attention) or token count (generation_attention)
// secondary: isl (context_attention) or context length (generation_attention)
func (t Lookup2D) Interp2D(primary, secondary float64) float64 {
	pAxis := t.primaryAxis()
	sAxis := t.secondaryAxis()
	if len(pAxis) == 0 || len(sAxis) == 0 {
		return 0
	}

	// Interpolate along secondary axis for a given primary index.
	interpAtPrimary := func(pi int) float64 {
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
		return interpAtPrimary(0)
	}
	if primary >= pAxis[len(pAxis)-1] {
		return interpAtPrimary(len(pAxis) - 1)
	}
	pi := sort.SearchFloat64s(pAxis, primary)
	lo := interpAtPrimary(pi - 1)
	hi := interpAtPrimary(pi)
	frac := (primary - pAxis[pi-1]) / (pAxis[pi] - pAxis[pi-1])
	return lo + frac*(hi-lo)
}

// clampPositive returns max(0, v) to prevent negative interpolation artifacts.
func clampPositive(v float64) float64 { return math.Max(0, v) }
