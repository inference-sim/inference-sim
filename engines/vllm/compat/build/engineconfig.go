// Moved from inference-sim/cmd/root.go at 63b211633e587a48a650922a2a2c2c2927199533.
// Moved from inference-sim/cmd/replay.go at 63b211633e587a48a650922a2a2c2c2927199533.

package build

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// loraConfigFile is the on-disk shape of a --lora-config YAML file: a single top-level lora:
// block matching contracts/config-schema.md. Strict-parsed (R10).
type loraConfigFile struct {
	LoRA engine.LoRAConfig `yaml:"lora"`
}

// LoadLoRAConfigFile parses a --lora-config YAML file's lora: block into an engine.LoRAConfig,
// with strict field checking (R10), so a typo never silently no-ops. The error text is the
// one the CLI has always printed.
func LoadLoRAConfigFile(path string) (engine.LoRAConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return engine.LoRAConfig{}, fmt.Errorf("Failed to read --lora-config file %q: %v", path, err) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	var f loraConfigFile
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&f); err != nil {
		return engine.LoRAConfig{}, fmt.Errorf("Failed to parse --lora-config file %q: %v", path, err) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	return f.LoRA, nil
}

// LoRADefaults are defaults.yaml's lora: cost coefficients, the fallback for fields neither
// the --lora-config file nor a flag sets. The caller reads defaults.yaml and hands them over.
type LoRADefaults struct {
	LoadBaseLatencyUs     float64
	LoadBandwidthBytesUs  float64
	FootprintBytesPerRank float64
	// StepOverheadTiers maps an adapter rank to its K6/K7 overhead coefficients.
	StepOverheadTiers map[int][2]float64
}

// LoRAFlags are the scalar --lora-* flags, each applied only when its Set is true (R18), so
// an unset flag defers to the file and defaults rather than clobbering them.
type LoRAFlags struct {
	AdapterCapacity          int
	AdapterCapacitySet       bool
	LoadBaseLatencyUs        float64
	LoadBaseLatencyUsSet     bool
	LoadBandwidthBytesUs     float64
	LoadBandwidthBytesUsSet  bool
	FootprintBytesPerRank    float64
	FootprintBytesPerRankSet bool
}

// ResolveLoRAConfig assembles the engine's LoRA configuration from three sources, in
// increasing precedence: defaults (cost-coefficient fallback, nil for none), the optional
// --lora-config file (path, "" for none; the adapter registry and any coefficients it sets),
// and the scalar flags. The result is validated; an invalid one is an error with the CLI's
// text (e.g. adapters declared with adapter_capacity 0).
//
// With no file and no flags set, the config still carries the defaults' cost coefficients but
// declares no adapters, so HasAdapters() is false and the subsystem is inert (INV-6).
func ResolveLoRAConfig(path string, defs *LoRADefaults, flags LoRAFlags) (engine.LoRAConfig, error) {
	var cfg engine.LoRAConfig
	if path != "" {
		c, err := LoadLoRAConfigFile(path)
		if err != nil {
			return engine.LoRAConfig{}, err
		}
		cfg = c
	}

	// defaults.yaml cost-coefficient fallback: fill only fields the file did not set, so an
	// unset flag defers to the file/defaults rather than clobbering it (R18).
	if defs != nil {
		if cfg.LoadBaseLatencyUs == nil {
			v := defs.LoadBaseLatencyUs
			cfg.LoadBaseLatencyUs = &v
		}
		if cfg.LoadBandwidthBytesUs == nil {
			v := defs.LoadBandwidthBytesUs
			cfg.LoadBandwidthBytesUs = &v
		}
		if cfg.FootprintBytesPerRank == nil {
			v := defs.FootprintBytesPerRank
			cfg.FootprintBytesPerRank = &v
		}
		if len(cfg.StepOverheadTiers) == 0 && len(defs.StepOverheadTiers) > 0 {
			cfg.StepOverheadTiers = make(map[int]engine.StepOverheadTier, len(defs.StepOverheadTiers))
			for rank, t := range defs.StepOverheadTiers {
				k6, k7 := t[0], t[1]
				cfg.StepOverheadTiers[rank] = engine.StepOverheadTier{K6: &k6, K7: &k7}
			}
		}
	}

	// Scalar flag overrides (R18: only when explicitly set).
	if flags.AdapterCapacitySet {
		v := flags.AdapterCapacity
		cfg.AdapterCapacity = &v
	}
	if flags.LoadBaseLatencyUsSet {
		v := flags.LoadBaseLatencyUs
		cfg.LoadBaseLatencyUs = &v
	}
	if flags.LoadBandwidthBytesUsSet {
		v := flags.LoadBandwidthBytesUs
		cfg.LoadBandwidthBytesUs = &v
	}
	if flags.FootprintBytesPerRankSet {
		v := flags.FootprintBytesPerRank
		cfg.FootprintBytesPerRank = &v
	}

	if err := cfg.Validate(); err != nil {
		return engine.LoRAConfig{}, fmt.Errorf("Invalid LoRA configuration: %v", err) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	return cfg, nil
}

// CheckPolicyNames refuses a scheduler or preemption policy the compatibility engine does not
// run, naming the valid ones, with the CLI's text. The scheduler is checked first.
func CheckPolicyNames(scheduler, preemption string) error {
	if !engine.IsValidScheduler(scheduler) {
		return fmt.Errorf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(namesOf(engine.SchedulerNameTable()), ", ")) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	if !engine.PreemptionPolicyNameTable()[preemption] {
		return fmt.Errorf("Unknown preemption policy %q. Valid: %s", preemption, strings.Join(namesOf(engine.PreemptionPolicyNameTable()), ", ")) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	return nil
}

// namesOf is a name table's non-empty names, sorted.
func namesOf(m map[string]bool) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		if k != "" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	return names
}

// EngineSettings are the engine-side settings of one run, as the caller resolved them from the
// scenario, the kernel and the command line. EngineConfigOf turns them into the engine's
// configuration; the caller adds what is the simulator's own (horizon, seed, front end, the
// latency model, metrics).
type EngineSettings struct {
	// KV cache.
	TotalKVBlocks, BlockSizeTokens, KVCPUBlocks int64
	KVOffloadThreshold, KVTransferBandwidth     float64
	KVTransferBaseLatency                       int64
	KVOffload                                   engine.KVOffloadConfig
	KVTransferTicksPerBlock                     int64
	SequenceState                               engine.KVSequenceState
	PrefixHit                                   engine.KVPrefixHit
	// Batch formation.
	MaxNumSeqs, MaxNumBatchedTokens, LongPrefillTokenThreshold int64
	PrefixCachingDisabled                                      bool
	// Model and hardware.
	Model                engine.ModelConfig
	ModelName, GPU       string
	TP, DP               int
	EnableExpertParallel bool
	MaxModelLen          int64
	// Policy.
	Scheduler, PreemptionPolicy string
	// LoRA and speculative decoding, already resolved.
	LoRA        engine.LoRAConfig
	Speculative engine.SpeculativeConfig
}

// EngineConfig is the engine's configuration for one run: the five sub-configs the engine is
// built from. Run and replay build it the same way (INV-13).
type EngineConfig struct {
	KVCache       engine.KVCacheConfig
	Batch         engine.BatchConfig
	ModelHardware engine.ModelHardwareConfig
	Policy        engine.PolicyConfig
	LoRA          engine.LoRAConfig
	Speculative   engine.SpeculativeConfig
}

// EngineConfigOf builds the engine's configuration from the run's settings, through each
// sub-config's canonical constructor (R4).
func EngineConfigOf(s EngineSettings) EngineConfig {
	return EngineConfig{
		KVCache: engine.NewKVCacheConfig(s.TotalKVBlocks, s.BlockSizeTokens, s.KVCPUBlocks,
			s.KVOffloadThreshold, s.KVTransferBandwidth, s.KVTransferBaseLatency,
			engine.WithKVOffload(s.KVOffload), engine.WithKVTransferTicksPerBlock(s.KVTransferTicksPerBlock),
			engine.WithSequenceState(s.SequenceState), engine.WithPrefixHit(s.PrefixHit)),
		Batch: BatchConfigOf(s),
		ModelHardware: engine.NewModelHardwareConfig(s.Model, s.ModelName, s.GPU, s.TP, s.DP,
			s.EnableExpertParallel, s.MaxModelLen),
		Policy:      engine.NewPolicyConfig(s.Scheduler, s.PreemptionPolicy),
		LoRA:        s.LoRA,
		Speculative: s.Speculative,
	}
}

// BatchConfigOf is the batch-formation part of EngineConfigOf alone: the settings' sequence and
// token limits, long-prefill threshold and prefix-caching toggle.
func BatchConfigOf(s EngineSettings) engine.BatchConfig {
	return engine.NewBatchConfig(s.MaxNumSeqs, s.MaxNumBatchedTokens, s.LongPrefillTokenThreshold,
		engine.WithPrefixCachingDisabled(s.PrefixCachingDisabled))
}

// SpeculativeConfigOf is the engine's speculative-decoding configuration for a scenario's K and
// draft method at the resolved acceptance rate (engine.NewSpeculativeConfig): a refusal for an
// invalid combination, such as a rate with nothing drafted.
func SpeculativeConfigOf(k int, acceptance float64, method string) (engine.SpeculativeConfig, error) {
	return engine.NewSpeculativeConfig(k, acceptance, method)
}

// maxSafeReservedBytes bounds the LoRA HBM reservation the KV budget can set aside. int64(x) is
// exact for every float64 strictly below 2^63, and float64(math.MaxInt64) rounds UP to 2^63, so
// the bound is the exactly representable 2^62, far above any real reservation (the cost model
// caps at 1e18).
const maxSafeReservedBytes = float64(int64(1) << 62)

// AdapterReservedBytes is the static LoRA HBM reservation, in bytes, the KV budget sets aside
// for a resolved config: the cost model's AdapterReservedBytes, through engine.BuildAdapterCost
// so its activation matches the engine's own wiring (R4). 0 when the subsystem is inert (no
// adapters, no capacity, or the LoRA package not linked), leaving KV capacity unchanged
// (INV-6). A malformed cost config, or a reservation outside the representable range, is an
// error with the CLI's text.
func AdapterReservedBytes(cfg engine.LoRAConfig) (int64, error) {
	ac, err := engine.BuildAdapterCost(engine.SimConfig{LoRAConfig: cfg})
	if err != nil {
		return 0, fmt.Errorf("Invalid LoRA configuration (HBM reservation): %v", err) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	if ac == nil {
		return 0, nil
	}
	reserved := ac.AdapterReservedBytes()
	if math.IsNaN(reserved) || math.IsInf(reserved, 0) || reserved < 0 || reserved >= maxSafeReservedBytes {
		return 0, fmt.Errorf("Invalid LoRA configuration (HBM reservation): %v bytes is outside the representable range", reserved) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	return int64(reserved), nil
}

// AdapterRegistryOf is the registry of a config's declared adapters, nil when it declares none
// (the subsystem is inert), for checking a workload's adapter references against it. An
// invalid registry is an error with the CLI's text.
func AdapterRegistryOf(cfg engine.LoRAConfig) (engine.AdapterRegistry, error) {
	if !cfg.HasAdapters() {
		return nil, nil
	}
	r, err := engine.AdapterRegistryConstructor()(cfg.Adapters)
	if err != nil {
		return nil, fmt.Errorf("Invalid LoRA adapter registry: %v", err) //nolint:staticcheck // the CLI's message, moved verbatim
	}
	return r, nil
}
