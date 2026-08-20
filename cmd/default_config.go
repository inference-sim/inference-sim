package cmd

import (
	"bytes"
	"fmt"
	"os"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// Workload describes a preset workload configuration in defaults.yaml.
type Workload struct {
	PrefixTokens      int `yaml:"prefix_tokens"`
	PromptTokensMean  int `yaml:"prompt_tokens"`
	PromptTokensStdev int `yaml:"prompt_tokens_stdev"`
	PromptTokensMin   int `yaml:"prompt_tokens_min"`
	PromptTokensMax   int `yaml:"prompt_tokens_max"`
	OutputTokensMean  int `yaml:"output_tokens"`
	OutputTokensStdev int `yaml:"output_tokens_stdev"`
	OutputTokensMin   int `yaml:"output_tokens_min"`
	OutputTokensMax   int `yaml:"output_tokens_max"`
}

// Config represents the full defaults.yaml structure.
// All top-level sections must be listed to satisfy KnownFields(true) strict parsing (R10).
type Config struct {
	Defaults               map[string]DefaultConfig `yaml:"defaults"`
	Version                string                   `yaml:"version"`
	Workloads              map[string]Workload      `yaml:"workloads"`
	TrainedPhysicsDefaults *TrainedPhysicsDefaults  `yaml:"trained_physics_coefficients,omitempty"`
	LoRADefaults           *LoRADefaults            `yaml:"lora,omitempty"`
	// KVOffloadDevices maps a device_class name to its bandwidth/latency physics for
	// the KV-offload config surface (H5, #1587). Shipped constants (operator input),
	// never fitted (BC-G4). Inert: only consulted when --kv-offload-config names a
	// device_class. A struct-field map (decode target, read-only after load) — not an
	// exported package var (R8-compatible, like Defaults/Workloads above).
	KVOffloadDevices map[string]KVOffloadDeviceDefaults `yaml:"kv_offload_devices,omitempty"`
}

// KVOffloadDeviceDefaults is one device_class's resolved physics for KV offload
// (H5, #1587). Bandwidths are bytes per microsecond; base_latency is microseconds.
// The three required fields describe the O_DIRECT regime (the historical default).
//
// #1581 adds an optional non-linear device model, all opt-in (a device that omits
// these fields resolves byte-identically to pre-#1581, INV-6):
//   - a queue-depth bandwidth ramp (saturation_queue_depth Qsat + single_transfer_
//     fraction f₁): effective bandwidth ramps from f₁·peak at in-service depth q=1
//     up to the peak (read/write_bandwidth) at q=Qsat, flat beyond;
//   - a relative latency jitter stddev (latency_jitter_stddev σ);
//   - a buffered-I/O regime (buffered_*), selected per tier by direct_io=false; any
//     absent buffered field falls back to the O_DIRECT value.
//
// Optional fields are pointers so "absent" is distinct from an explicit zero (R9).
type KVOffloadDeviceDefaults struct {
	ReadBandwidth  float64 `yaml:"read_bandwidth"`
	WriteBandwidth float64 `yaml:"write_bandwidth"`
	BaseLatency    float64 `yaml:"base_latency"`

	// O_DIRECT regime device model (optional; absent => no ramp / no jitter).
	SaturationQueueDepth   *int64   `yaml:"saturation_queue_depth,omitempty"`
	SingleTransferFraction *float64 `yaml:"single_transfer_fraction,omitempty"`
	LatencyJitterStddev    *float64 `yaml:"latency_jitter_stddev,omitempty"`

	// Buffered-I/O regime (optional; each absent field falls back to O_DIRECT).
	BufferedReadBandwidth          *float64 `yaml:"buffered_read_bandwidth,omitempty"`
	BufferedWriteBandwidth         *float64 `yaml:"buffered_write_bandwidth,omitempty"`
	BufferedBaseLatency            *float64 `yaml:"buffered_base_latency,omitempty"`
	BufferedSaturationQueueDepth   *int64   `yaml:"buffered_saturation_queue_depth,omitempty"`
	BufferedSingleTransferFraction *float64 `yaml:"buffered_single_transfer_fraction,omitempty"`
	BufferedLatencyJitterStddev    *float64 `yaml:"buffered_latency_jitter_stddev,omitempty"`
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

// Define the inner structure for default config given model
type DefaultConfig struct {
	GPU               string `yaml:"GPU"`
	TensorParallelism int    `yaml:"tensor_parallelism"`
	HFRepo            string `yaml:"hf_repo,omitempty"`
}

func GetDefaultSpecs(LLM string) (GPU string, TensorParallelism int) {
	data, err := os.ReadFile(defaultsFilePath)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file: %v", err)
	}

	// Parse YAML with strict field checking (R10: typos must cause errors)
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
	}

	if _, modelExists := cfg.Defaults[LLM]; modelExists {
		return cfg.Defaults[LLM].GPU, cfg.Defaults[LLM].TensorParallelism
	} else {
		return "", 0
	}
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

// GetHFRepo returns the HuggingFace repository path for the given model from defaults.yaml.
// Returns ("", nil) if the model exists but has no hf_repo mapping.
// Returns ("", error) if the defaults file cannot be read or parsed (R1: no silent data loss).
func GetHFRepo(modelName string, defaultsFile string) (string, error) {
	data, err := os.ReadFile(defaultsFile)
	if err != nil {
		return "", fmt.Errorf("read defaults file %s: %w", defaultsFile, err)
	}
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return "", fmt.Errorf("parse defaults YAML: %w", err)
	}

	if dc, ok := cfg.Defaults[modelName]; ok {
		return dc.HFRepo, nil
	}
	return "", nil
}

