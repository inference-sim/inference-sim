package sim

import (
	"fmt"
	"math"
)

// KVOffloadConfig is the resolved, validated KV-cache offload configuration
// captured from vLLM's offload CLI/config surface (H5, issue #1587). It is a
// sub-config of KVCacheConfig (R16): grouped by module, independently validatable.
//
// SCOPE: this is the config *surface* only. In this PR the offload subsystem is
// INERT — no mechanism reads these fields (the consuming physics live in sibling
// holes #1588/#1589 and later). The zero value has Enabled=false, so a run without
// --kv-offload-config is byte-identical to a build without this feature (BC-G5,
// INV-6): sim/kv never reads Offload.
//
// The resolved struct carries fully-derived values (device classes already resolved
// to bandwidth/latency numbers, all defaults applied); it is what round-trips through
// the trace header (BC-G6). It is read-only after construction (BC-G4, never fitted):
// operator input, structurally separate from trained coefficients.
type KVOffloadConfig struct {
	// Enabled is true iff a kv_offload: block was supplied. The zero value (false)
	// is inert. All other fields are meaningful only when Enabled is true.
	Enabled bool

	// CPUBytesToUse is vLLM's cpu_bytes_to_use: CPU-tier capacity in bytes. Required
	// (>0) when Enabled — vLLM marks it required with no default. Interpreted as the
	// per-GPU (per-TP-rank) host CPU budget: the CPU-tier block capacity is
	// CPUBytesToUse / PerBlockBytes, both per-rank quantities.
	CPUBytesToUse int64

	// PerBlockBytes is the resolved KV-cache byte size of one GPU block on one TP
	// rank: KVBytesPerToken(model, tp) × BlockSizeTokens (H1, #1590). It is DERIVED
	// from the model + block size, not a user knob — sim/kv cannot import sim/latency
	// to compute it, so cmd resolves it and records it here so the tier-chain
	// mechanism can convert CPUBytesToUse into a block capacity and size transfer
	// jobs. Required (>0) when Enabled. Being derived, it is authoritative from the
	// trace header on replay and excluded from the flag-vs-header reconcile.
	PerBlockBytes int64

	// BlockSize is the offload block size in TOKENS. vLLM default = GPU block size.
	// It is an alternate encoding of BlocksPerChunk (BlockSize = BlocksPerChunk ×
	// gpu_block_size); the resolver keeps the two consistent, so both are always set.
	BlockSize int64

	// BlocksPerChunk is vLLM's blocks_per_chunk: GPU blocks coalesced into one offload
	// chunk (job batching width, #1588 BC-S5). vLLM default 1.
	BlocksPerChunk int64

	// TokensPerHash is vLLM's tokens_per_hash: the block-hash key stride (#1589 BC-K4).
	// vLLM makes it required with no default; BLIS defaults it to the GPU block size
	// (the natural one-hash-per-block stride).
	TokensPerHash int64

	// EvictionPolicy is vLLM's eviction_policy: which block is dropped. vLLM default
	// "lru"; "arc" also accepted.
	EvictionPolicy string

	// OffloadPromptOnly is vLLM's offload_prompt_only. vLLM DEFAULT TRUE (trap 1):
	// decode KV is NOT offloaded unless explicitly disabled.
	OffloadPromptOnly bool

	// SelfDescribingKVEvents is vLLM's self_describing_kv_events (observability only).
	// vLLM default false.
	SelfDescribingKVEvents bool

	// Tiers are the ordered secondary offload tiers (vLLM secondary_tiers). Tier 0 is
	// consulted before tier 1, etc. vLLM default empty (single CPU tier, no spill).
	// A slice (not a map, R8) so ordering is deterministic (INV-6).
	Tiers []KVOffloadTier
}

// KVOffloadTier is one resolved secondary offload tier. Only the filesystem ("fs")
// tier type is representable in BLIS today; the resolver rejects obj/p2p/example
// loudly (BC-G1). Bandwidths are per-(tier, direction) — never a single number
// (#1588 BC-S3).
type KVOffloadTier struct {
	// Type is the tier kind. Only "fs" is accepted; obj/p2p/example are rejected at
	// resolution (BC-G1 fails-loudly branch).
	Type string

	// RootDir is the fs tier's device directory (vLLM root_dir), required for "fs".
	RootDir string

	// NReadThreads / NWriteThreads are vLLM's per-tier read/write server counts
	// (#1588 BC-S1/S2). vLLM default 16 each.
	NReadThreads  int64
	NWriteThreads int64

	// Locality is vLLM's locality: "" (unset), "LOCAL", or "REMOTE". Latency class.
	Locality string

	// EnableKVEvents is vLLM's per-tier enable_kv_events. vLLM default false.
	EnableKVEvents bool

	// DirectIO makes vLLM's runtime O_DIRECT probe an EXPLICIT config axis: vLLM
	// discovers O_DIRECT support at startup and silently falls back to buffered I/O,
	// but a simulator cannot probe the operator's disk and the two modes have
	// substantially different physics. BLIS requires it to be declared per fs tier
	// (enforced in the resolver) and records it, since a buffered-I/O trace is not
	// comparable to a direct-I/O one. BLIS-only; no vLLM knob.
	DirectIO bool

	// DeviceClass is a BLIS-only informational echo of the device_class that resolved
	// the bandwidth/latency triple. The resolved numbers below are authoritative on
	// replay (never re-resolved against the replay host's defaults.yaml — cross-host
	// INV-13).
	DeviceClass string

	// ReadBandwidth / WriteBandwidth are the resolved per-direction bandwidths in
	// bytes per microsecond. BaseLatency is the fixed per-access latency in
	// microseconds.
	ReadBandwidth  float64
	WriteBandwidth float64
	BaseLatency    float64
}

// Valid enumerations for the offload config surface.
const (
	kvOffloadEvictionLRU = "lru"
	kvOffloadEvictionARC = "arc"

	kvOffloadTierFS = "fs"

	kvOffloadLocalityLocal  = "LOCAL"
	kvOffloadLocalityRemote = "REMOTE"
)

// IsEnabled reports whether the offload subsystem is configured. A zero-value
// KVOffloadConfig is inert (BC-G5).
func (c KVOffloadConfig) IsEnabled() bool { return c.Enabled }

// Validate checks the invariants of a RESOLVED KVOffloadConfig (BC-G3). It is a
// no-op when the config is inert. It returns an error (never panics — R6) naming the
// offending field; the CLI boundary turns that into logrus.Fatalf.
//
// Validate operates on the already-resolved struct: it does NOT re-check the
// block_size / blocks_per_chunk mutual-exclusion (that is a user-input check in the
// resolver — both fields are always present and consistent here), and it does NOT
// check that direct_io was explicitly supplied (a resolved bool has no "unset" state;
// the resolver enforces the explicit-declaration requirement). Float fields are
// guarded against NaN/Inf, mirroring NewKVCacheConfig and sim/kv/tiered.go.
func (c KVOffloadConfig) Validate() error {
	if !c.Enabled {
		return nil
	}
	if c.CPUBytesToUse <= 0 {
		return fmt.Errorf("kv_offload: cpu_bytes_to_use must be > 0 when offload is enabled, got %d", c.CPUBytesToUse)
	}
	// NOTE: PerBlockBytes is NOT checked here. It is a model-derived value resolved
	// downstream (cmd, where the model + TP are known), so it is legitimately 0 at
	// config-surface validation time. The tier-chain factory (NewOffloadCache)
	// enforces PerBlockBytes > 0 at the point the mechanism consumes it (#1590).
	if c.BlockSize <= 0 {
		return fmt.Errorf("kv_offload: block_size must be > 0, got %d", c.BlockSize)
	}
	if c.BlocksPerChunk <= 0 {
		return fmt.Errorf("kv_offload: blocks_per_chunk must be > 0, got %d", c.BlocksPerChunk)
	}
	if c.TokensPerHash <= 0 {
		return fmt.Errorf("kv_offload: tokens_per_hash must be > 0, got %d", c.TokensPerHash)
	}
	if c.EvictionPolicy != kvOffloadEvictionLRU && c.EvictionPolicy != kvOffloadEvictionARC {
		return fmt.Errorf("kv_offload: eviction_policy must be %q or %q, got %q", kvOffloadEvictionLRU, kvOffloadEvictionARC, c.EvictionPolicy)
	}
	for i, t := range c.Tiers {
		if err := t.validate(i); err != nil {
			return err
		}
	}
	return nil
}

// validate checks one resolved tier. index is used only for error messages.
func (t KVOffloadTier) validate(index int) error {
	if t.Type != kvOffloadTierFS {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].type must be %q (only the filesystem tier is representable in BLIS; obj/p2p/example are not yet supported), got %q", index, kvOffloadTierFS, t.Type)
	}
	if t.RootDir == "" {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].root_dir is required for an %q tier", index, kvOffloadTierFS)
	}
	if t.Locality != "" && t.Locality != kvOffloadLocalityLocal && t.Locality != kvOffloadLocalityRemote {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].locality must be %q or %q (or unset), got %q", index, kvOffloadLocalityLocal, kvOffloadLocalityRemote, t.Locality)
	}
	if t.NReadThreads <= 0 {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].n_read_threads must be > 0, got %d", index, t.NReadThreads)
	}
	if t.NWriteThreads <= 0 {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].n_write_threads must be > 0, got %d", index, t.NWriteThreads)
	}
	if err := validatePositiveFinite(t.ReadBandwidth, index, "read_bandwidth"); err != nil {
		return err
	}
	if err := validatePositiveFinite(t.WriteBandwidth, index, "write_bandwidth"); err != nil {
		return err
	}
	if math.IsNaN(t.BaseLatency) || math.IsInf(t.BaseLatency, 0) || t.BaseLatency < 0 {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].base_latency must be a finite value >= 0, got %v", index, t.BaseLatency)
	}
	return nil
}

// validatePositiveFinite rejects a non-finite or non-positive bandwidth, naming the field.
func validatePositiveFinite(v float64, index int, field string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return fmt.Errorf("kv_offload: secondary_tiers[%d].%s must be a finite value > 0, got %v", index, field, v)
	}
	return nil
}
