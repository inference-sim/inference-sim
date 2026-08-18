package cmd

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// kvOffloadConfigPath backs the single new --kv-offload-config flag (H5, #1587).
// Absent ("") ⇒ the offload subsystem is inert and output is byte-identical to a
// build without the feature (BC-G5). Registered on run and replay via the shared
// registerSimConfigFlags (INV-13 parity).
var kvOffloadConfigPath string

// kvOffloadFile is the on-disk shape of a --kv-offload-config YAML file: a single
// top-level kv_offload: block. Strict-parsed (R10). Mirrors the --lora-config /
// --saturation-config precedent.
type kvOffloadFile struct {
	KVOffload *kvOffloadBlock `yaml:"kv_offload"`
}

// kvOffloadBlock is the user-facing kv_offload: block. All fields are pointers so an
// absent key is distinguishable from an explicit zero (R9): the resolver applies
// vLLM's default only when the pointer is nil.
//
// NOTE store_threshold: it is captured (as *int) but is NOT a live knob on the
// multi-tier path — vLLM's TieringOffloadingSpec rejects values >= 2 (it is
// single-tier-only). The resolver rejects >= 2 loudly (BC-G1) and treats nil/0/1 as a
// no-op (matching vLLM, which accepts 0/1 and ignores them on this path).
type kvOffloadBlock struct {
	CPUBytesToUse          *int64               `yaml:"cpu_bytes_to_use"`
	BlockSize              *int64               `yaml:"block_size"`
	BlocksPerChunk         *int64               `yaml:"blocks_per_chunk"`
	TokensPerHash          *int64               `yaml:"tokens_per_hash"`
	EvictionPolicy         *string              `yaml:"eviction_policy"`
	OffloadPromptOnly      *bool                `yaml:"offload_prompt_only"`
	SelfDescribingKVEvents *bool                `yaml:"self_describing_kv_events"`
	StoreThreshold         *int                 `yaml:"store_threshold"`
	SecondaryTiers         []kvOffloadTierBlock `yaml:"secondary_tiers"`
}

// kvOffloadTierBlock is one user-facing secondary tier. Pointer fields (R9).
type kvOffloadTierBlock struct {
	Type           *string  `yaml:"type"`
	RootDir        *string  `yaml:"root_dir"`
	NReadThreads   *int64   `yaml:"n_read_threads"`
	NWriteThreads  *int64   `yaml:"n_write_threads"`
	Locality       *string  `yaml:"locality"`
	EnableKVEvents *bool    `yaml:"enable_kv_events"`
	DirectIO       *bool    `yaml:"direct_io"`
	DeviceClass    *string  `yaml:"device_class"`
	ReadBandwidth  *float64 `yaml:"read_bandwidth"`
	WriteBandwidth *float64 `yaml:"write_bandwidth"`
	BaseLatency    *float64 `yaml:"base_latency"`
}

// loadKVOffloadConfigFile reads and strictly parses a --kv-offload-config file.
// Returns an error (never Fatalf — so it is unit-testable/fuzzable, Deviation #9);
// the CLI wrapper turns errors into logrus.Fatalf. Unknown keys error via
// KnownFields(true) (R10).
func loadKVOffloadConfigFile(path string) (kvOffloadFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return kvOffloadFile{}, fmt.Errorf("read --kv-offload-config file %q: %w", path, err)
	}
	f, err := parseKVOffloadBytes(data)
	if err != nil {
		return f, fmt.Errorf("parse --kv-offload-config file %q: %w", path, err)
	}
	return f, nil
}

// parseKVOffloadBytes strictly decodes kv_offload YAML bytes (R10). Split out so it is
// directly fuzzable without touching disk.
func parseKVOffloadBytes(data []byte) (kvOffloadFile, error) {
	var f kvOffloadFile
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&f); err != nil {
		return f, err
	}
	return f, nil
}

// resolveKVOffload turns a parsed kv_offload block into a resolved, validated
// sim.KVOffloadConfig. It is PURE and error-returning (Deviation #9) so every reject
// branch is unit-testable and fuzzable. It applies vLLM's defaults knob-for-knob
// (BC-G2), resolves device_class against the shipped device physics (explicit
// read/write/base override the class), enforces the block_size XOR blocks_per_chunk
// user-input rule then derives the canonical pair, rejects store_threshold>=2 and
// non-fs tiers loudly (BC-G1), and finally runs cfg.Validate() (BC-G3).
//
// gpuBlockSizeTokens is the GPU block size (--block-size-in-tokens); vLLM's block_size
// default equals it, and it converts between the block_size and blocks_per_chunk
// encodings of the same quantity.
func resolveKVOffload(block *kvOffloadBlock, devices map[string]KVOffloadDeviceDefaults, gpuBlockSizeTokens int64) (sim.KVOffloadConfig, error) {
	if block == nil {
		return sim.KVOffloadConfig{}, fmt.Errorf("kv_offload: the --kv-offload-config file has no top-level kv_offload: block")
	}
	if gpuBlockSizeTokens <= 0 {
		return sim.KVOffloadConfig{}, fmt.Errorf("kv_offload: GPU block size (--block-size-in-tokens) must be > 0, got %d", gpuBlockSizeTokens)
	}
	cfg := sim.KVOffloadConfig{Enabled: true}

	// cpu_bytes_to_use: required when the block is present (vLLM marks it required).
	if block.CPUBytesToUse == nil {
		return cfg, fmt.Errorf("kv_offload: cpu_bytes_to_use is required")
	}
	cfg.CPUBytesToUse = *block.CPUBytesToUse

	// store_threshold: reject < 0 (invalid) and >= 2 (vLLM TieringOffloadingSpec
	// rejects it); 0/1 are accepted as a no-op on the multi-tier path (vLLM parity).
	if block.StoreThreshold != nil {
		if *block.StoreThreshold < 0 {
			return cfg, fmt.Errorf("kv_offload: store_threshold must be >= 0, got %d", *block.StoreThreshold)
		}
		if *block.StoreThreshold >= 2 {
			return cfg, fmt.Errorf("kv_offload: store_threshold=%d: store_threshold is not supported for TieringOffloadingSpec (values >= 2 are rejected; it is single-tier-only and fixed at 1 on the multi-tier offload path)", *block.StoreThreshold)
		}
	}

	// block_size XOR blocks_per_chunk (mutually exclusive alternate encodings of the
	// same chunk-granularity quantity). Derive one canonical pair.
	if block.BlockSize != nil && block.BlocksPerChunk != nil {
		return cfg, fmt.Errorf("kv_offload: block_size and blocks_per_chunk are mutually exclusive (they encode the same quantity: block_size = blocks_per_chunk × gpu_block_size); set at most one")
	}
	switch {
	case block.BlockSize != nil:
		bs := *block.BlockSize
		if bs <= 0 || bs%gpuBlockSizeTokens != 0 {
			return cfg, fmt.Errorf("kv_offload: block_size (%d) must be a positive multiple of the GPU block size (%d)", bs, gpuBlockSizeTokens)
		}
		cfg.BlockSize = bs
		cfg.BlocksPerChunk = bs / gpuBlockSizeTokens
	case block.BlocksPerChunk != nil:
		bpc := *block.BlocksPerChunk
		if bpc <= 0 {
			return cfg, fmt.Errorf("kv_offload: blocks_per_chunk must be > 0, got %d", bpc)
		}
		cfg.BlocksPerChunk = bpc
		cfg.BlockSize = bpc * gpuBlockSizeTokens
		if cfg.BlockSize/gpuBlockSizeTokens != bpc { // int64 overflow guard
			return cfg, fmt.Errorf("kv_offload: blocks_per_chunk (%d) is too large — block_size = blocks_per_chunk × gpu_block_size (%d) would overflow int64", bpc, gpuBlockSizeTokens)
		}
	default:
		cfg.BlocksPerChunk = 1             // vLLM default
		cfg.BlockSize = gpuBlockSizeTokens // vLLM default = GPU block size
	}

	// tokens_per_hash: vLLM has no default (required); BLIS defaults to the GPU block
	// size (natural one-hash-per-block stride).
	if block.TokensPerHash != nil {
		if *block.TokensPerHash <= 0 {
			return cfg, fmt.Errorf("kv_offload: tokens_per_hash must be > 0, got %d", *block.TokensPerHash)
		}
		cfg.TokensPerHash = *block.TokensPerHash
	} else {
		cfg.TokensPerHash = gpuBlockSizeTokens
	}

	// eviction_policy: vLLM default "lru".
	cfg.EvictionPolicy = "lru"
	if block.EvictionPolicy != nil {
		cfg.EvictionPolicy = *block.EvictionPolicy
	}

	// offload_prompt_only: vLLM DEFAULT TRUE (trap 1).
	cfg.OffloadPromptOnly = true
	if block.OffloadPromptOnly != nil {
		cfg.OffloadPromptOnly = *block.OffloadPromptOnly
	}

	// self_describing_kv_events: vLLM default false.
	if block.SelfDescribingKVEvents != nil {
		cfg.SelfDescribingKVEvents = *block.SelfDescribingKVEvents
	}

	// secondary_tiers: vLLM default empty.
	for i, tb := range block.SecondaryTiers {
		tier, err := resolveKVOffloadTier(i, tb, devices)
		if err != nil {
			return cfg, err
		}
		cfg.Tiers = append(cfg.Tiers, tier)
	}

	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// resolveKVOffloadTier resolves one secondary tier. index is used in error messages.
func resolveKVOffloadTier(index int, tb kvOffloadTierBlock, devices map[string]KVOffloadDeviceDefaults) (sim.KVOffloadTier, error) {
	var tier sim.KVOffloadTier

	// type: required; only "fs" is representable (reject obj/p2p/example loudly, BC-G1).
	if tb.Type == nil {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].type is required", index)
	}
	tier.Type = *tb.Type
	if tier.Type != "fs" {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].type=%q is not supported in BLIS yet (only \"fs\" is representable; obj/p2p/example tiers have no faithful config mapping — see #1587/#1585)", index, tier.Type)
	}

	// root_dir: required for fs.
	if tb.RootDir == nil || *tb.RootDir == "" {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].root_dir is required for an \"fs\" tier", index)
	}
	tier.RootDir = *tb.RootDir

	// n_read_threads / n_write_threads: vLLM default 16 each.
	tier.NReadThreads = 16
	if tb.NReadThreads != nil {
		tier.NReadThreads = *tb.NReadThreads
	}
	tier.NWriteThreads = 16
	if tb.NWriteThreads != nil {
		tier.NWriteThreads = *tb.NWriteThreads
	}

	// locality: default unset ("").
	if tb.Locality != nil {
		tier.Locality = *tb.Locality
	}

	// enable_kv_events: vLLM default false.
	if tb.EnableKVEvents != nil {
		tier.EnableKVEvents = *tb.EnableKVEvents
	}

	// direct_io: REQUIRED — BLIS makes vLLM's runtime O_DIRECT probe an explicit
	// config axis (direct vs buffered I/O are different physics regimes; a simulator
	// cannot probe the operator's disk). No silent default.
	if tb.DirectIO == nil {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].direct_io must be set explicitly (BLIS makes vLLM's runtime O_DIRECT probe an explicit config axis; direct vs buffered I/O are materially different storage physics)", index)
	}
	tier.DirectIO = *tb.DirectIO

	// bandwidth/latency: a device_class resolves the triple; explicit fields override.
	hasClass := tb.DeviceClass != nil
	if hasClass {
		dev, ok := devices[*tb.DeviceClass]
		if !ok {
			return tier, fmt.Errorf("kv_offload: secondary_tiers[%d].device_class=%q is not defined in defaults.yaml kv_offload_devices (known: %s)", index, *tb.DeviceClass, knownDeviceClasses(devices))
		}
		tier.DeviceClass = *tb.DeviceClass
		tier.ReadBandwidth = dev.ReadBandwidth
		tier.WriteBandwidth = dev.WriteBandwidth
		tier.BaseLatency = dev.BaseLatency
	}
	if tb.ReadBandwidth != nil {
		tier.ReadBandwidth = *tb.ReadBandwidth
	}
	if tb.WriteBandwidth != nil {
		tier.WriteBandwidth = *tb.WriteBandwidth
	}
	if tb.BaseLatency != nil {
		tier.BaseLatency = *tb.BaseLatency
	}
	// Every tier needs a resolvable physics triple: either a device_class or the full
	// explicit read_bandwidth + write_bandwidth + base_latency (per-(tier,direction)
	// bandwidth is required — never a single number, #1588 BC-S3).
	if !hasClass && (tb.ReadBandwidth == nil || tb.WriteBandwidth == nil || tb.BaseLatency == nil) {
		return tier, fmt.Errorf("kv_offload: secondary_tiers[%d] needs a device_class OR an explicit read_bandwidth + write_bandwidth + base_latency triple", index)
	}
	return tier, nil
}

// knownDeviceClasses returns the sorted device_class names for a deterministic error
// message (INV-6: no map-iteration order in output).
func knownDeviceClasses(devices map[string]KVOffloadDeviceDefaults) string {
	if len(devices) == 0 {
		return "<none configured>"
	}
	names := make([]string, 0, len(devices))
	for n := range devices {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// resolveKVOffloadConfig is the thin CLI wrapper (Deviation #9): it reads the flag,
// loads the file + shipped device physics, calls the pure resolver, and logrus.Fatalf's
// on any error (CLI boundary). Absent flag ⇒ inert zero value. Single construction
// site (R4), shared by run and replay for INV-13 parity.
func resolveKVOffloadConfig(cmd *cobra.Command) sim.KVOffloadConfig {
	if kvOffloadConfigPath == "" {
		return sim.KVOffloadConfig{}
	}
	f, err := loadKVOffloadConfigFile(kvOffloadConfigPath)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	devices := loadDefaultsConfig(defaultsFilePath).KVOffloadDevices
	cfg, err := resolveKVOffload(f.KVOffload, devices, blockSizeTokens)
	if err != nil {
		logrus.Fatalf("--kv-offload-config %q: %v", kvOffloadConfigPath, err)
	}
	return cfg
}

// simToHeaderOffload converts a resolved sim.KVOffloadConfig into its trace-header
// serialization (H5, #1587). Returns nil when inert, so a disabled run omits the
// kv_offload header key entirely (BC-G5). Lossless inverse of headerToSimOffload.
func simToHeaderOffload(c sim.KVOffloadConfig) *workload.TraceKVOffloadConfig {
	if !c.IsEnabled() {
		return nil
	}
	h := &workload.TraceKVOffloadConfig{
		CPUBytesToUse:          c.CPUBytesToUse,
		BlockSize:              c.BlockSize,
		BlocksPerChunk:         c.BlocksPerChunk,
		TokensPerHash:          c.TokensPerHash,
		EvictionPolicy:         c.EvictionPolicy,
		OffloadPromptOnly:      c.OffloadPromptOnly,
		SelfDescribingKVEvents: c.SelfDescribingKVEvents,
	}
	for _, t := range c.Tiers {
		h.Tiers = append(h.Tiers, workload.TraceKVOffloadTier{
			Type:           t.Type,
			RootDir:        t.RootDir,
			NReadThreads:   t.NReadThreads,
			NWriteThreads:  t.NWriteThreads,
			Locality:       t.Locality,
			EnableKVEvents: t.EnableKVEvents,
			DirectIO:       t.DirectIO,
			DeviceClass:    t.DeviceClass,
			ReadBandwidth:  t.ReadBandwidth,
			WriteBandwidth: t.WriteBandwidth,
			BaseLatency:    t.BaseLatency,
		})
	}
	return h
}

// reconcileReplayKVOffload resolves the KV-offload config for blis replay. Unlike
// --lora-config (flags-only on replay), the offload config round-trips through the
// trace header, so on replay the HEADER is authoritative (BC-G6):
//   - a recorded config that this binary cannot reconstruct/validate ⇒ logrus.Fatalf
//     (INV-13, never a silent degrade to single-tier);
//   - a --kv-offload-config flag is accepted only if it resolves identical to the
//     header (INV-13 "identical flags"); a genuine mismatch ⇒ Fatalf;
//   - a flag that would ADD offload to a trace captured without it ⇒ Fatalf.
//
// Returns the inert zero value when the trace carries no offload config.
func reconcileReplayKVOffload(cmd *cobra.Command, headerBlock *workload.TraceKVOffloadConfig) sim.KVOffloadConfig {
	flagChanged := cmd.Flags().Changed("kv-offload-config")
	var flagCfg sim.KVOffloadConfig
	if flagChanged {
		// resolveKVOffloadConfig itself logrus.Fatalf's on a malformed flag file.
		flagCfg = resolveKVOffloadConfig(cmd)
	}
	cfg, err := resolveReplayKVOffload(headerBlock, flagChanged, flagCfg)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// resolveReplayKVOffload is the PURE, error-returning core of replay's
// header-authoritative reconciliation (Deviation #9 — every reject branch is
// unit-testable). See reconcileReplayKVOffload for the policy summary.
func resolveReplayKVOffload(headerBlock *workload.TraceKVOffloadConfig, flagChanged bool, flagCfg sim.KVOffloadConfig) (sim.KVOffloadConfig, error) {
	headerOffload := headerToSimOffload(headerBlock)
	if headerOffload.IsEnabled() {
		if err := headerOffload.Validate(); err != nil {
			return sim.KVOffloadConfig{}, fmt.Errorf("blis replay cannot reproduce the trace's kv_offload config: %w (INV-13: never silent degradation)", err)
		}
		if flagChanged && !reflect.DeepEqual(flagCfg, headerOffload) {
			return sim.KVOffloadConfig{}, fmt.Errorf("--kv-offload-config conflicts with the kv_offload config recorded in the trace header; on replay the header is authoritative — omit the flag or pass the identical config")
		}
		return headerOffload, nil
	}
	if flagChanged && flagCfg.IsEnabled() {
		return sim.KVOffloadConfig{}, fmt.Errorf("the trace was captured without a kv_offload config; --kv-offload-config cannot add one on replay (INV-13)")
	}
	return sim.KVOffloadConfig{}, nil
}

// headerToSimOffload reconstructs a resolved sim.KVOffloadConfig from a trace header
// (BC-G6). Returns the inert zero value when the header carries no offload block.
// Lossless inverse of simToHeaderOffload. The caller (replay) runs Validate() on the
// result and logrus.Fatalf's if the recorded config cannot be reproduced.
func headerToSimOffload(h *workload.TraceKVOffloadConfig) sim.KVOffloadConfig {
	if h == nil {
		return sim.KVOffloadConfig{}
	}
	c := sim.KVOffloadConfig{
		Enabled:                true,
		CPUBytesToUse:          h.CPUBytesToUse,
		BlockSize:              h.BlockSize,
		BlocksPerChunk:         h.BlocksPerChunk,
		TokensPerHash:          h.TokensPerHash,
		EvictionPolicy:         h.EvictionPolicy,
		OffloadPromptOnly:      h.OffloadPromptOnly,
		SelfDescribingKVEvents: h.SelfDescribingKVEvents,
	}
	for _, t := range h.Tiers {
		c.Tiers = append(c.Tiers, sim.KVOffloadTier{
			Type:           t.Type,
			RootDir:        t.RootDir,
			NReadThreads:   t.NReadThreads,
			NWriteThreads:  t.NWriteThreads,
			Locality:       t.Locality,
			EnableKVEvents: t.EnableKVEvents,
			DirectIO:       t.DirectIO,
			DeviceClass:    t.DeviceClass,
			ReadBandwidth:  t.ReadBandwidth,
			WriteBandwidth: t.WriteBandwidth,
			BaseLatency:    t.BaseLatency,
		})
	}
	return c
}
