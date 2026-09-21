package cmd

import (
	"bytes"
	"os"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// Config represents the full defaults.yaml structure.
// All top-level sections must be listed to satisfy KnownFields(true) strict parsing (R10).
//
// #1768: there is deliberately NO `defaults:` section. It held per-model GPU /
// tensor_parallelism / hf_repo, which NS-6 (#1733) made unreachable on every run path — the
// deployment is a required operator input (--hardware/--tp, see requireDeploymentFlags) and
// the model config comes from the catalog (--catalog / BLIS_CATALOG, #1731). Because
// KnownFields(true) is one-way — an undeclared YAML key is a hard error, an unsupplied Go
// field is legal and zero-valued — a `defaults:` block surviving in a hand-maintained copy of
// the file is now refused at load rather than silently ignored. Do not re-add the field to
// accept such a file: per-model deployment policy has no consumer to be silent about.
//
// #1769: there is likewise NO `workloads:` section. The named presets (chatbot,
// summarization, contentgen, multidoc) existed here AND in the catalog's workloads/ namespace
// with nothing keeping the two copies in sync; the catalog is now the single source of truth
// (see cmd/catalog_workloads.go), and by the same one-way KnownFields(true) rule a surviving
// `workloads:` block is refused at load rather than parsed and ignored.
//
// #1770: for the same reason there is deliberately NO `kv_offload_devices:` section. The
// KV-offload storage-device physics table was duplicated between this file and the catalog's
// devices/storage.yaml with nothing keeping the copies in sync; the catalog is now the single
// source of truth (see cmd/catalog_devices.go). A surviving kv_offload_devices: block in a
// hand-maintained copy of this file is likewise refused at load — the same one-way
// KnownFields(true) consequence, and the same reason not to re-declare the field.
type Config struct {
	Version                string                  `yaml:"version"`
	TrainedPhysicsDefaults *TrainedPhysicsDefaults `yaml:"trained_physics_coefficients,omitempty"`
	LoRADefaults           *LoRADefaults           `yaml:"lora,omitempty"`
}

// LoRADefaults holds inert defaults for the LoRA control-plane subsystem's cost
// coefficients. Present in defaults.yaml but only applied to a run when adapters are
// configured (INV-6 no-op default). These values seed the --lora-* flag defaults;
// they are NOT the adapter registry (registry is declared per-run via a config file).
type LoRADefaults struct {
	LoadBaseLatencyUs     float64                          `yaml:"load_base_latency_us"`
	LoadBandwidthBytesUs  float64                          `yaml:"load_bandwidth_bytes_us"`
	FootprintBytesPerRank float64                          `yaml:"footprint_bytes_per_rank"`
	StepOverheadTiers     map[int]LoRAStepOverheadDefaults `yaml:"step_overhead_tiers,omitempty"`
}

// LoRAStepOverheadDefaults mirrors sim.StepOverheadTier for defaults.yaml parsing.
type LoRAStepOverheadDefaults struct {
	K6 float64 `yaml:"k6"`
	K7 float64 `yaml:"k7"`
}

// TrainedPhysicsDefaults holds physics-informed roofline + learned correction coefficients.
// AlphaCoeffs has 3 elements (α₀-α₂): API/framework overheads in µs.
// BetaCoeffs has 11 elements (β₁-β₁₀ + β_EP): roofline corrections and per-component overheads.
// Trained from iter29 (sequential golden section search, β₆ +57%, loss 34.57%).
type TrainedPhysicsDefaults struct {
	AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
	BetaCoeffs  []float64 `yaml:"beta_coeffs"`
}

// loadDefaultsConfig parses defaults.yaml into a Config struct.
// Uses strict field checking (R10).
func loadDefaultsConfig(path string) Config {
	data, err := os.ReadFile(path)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file: %v", err)
	}
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
	}
	return cfg
}
