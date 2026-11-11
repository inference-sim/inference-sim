package coefficient

// The coefficients that price a transfer to or from an offload tier (design SUB-18, SUB-3).
//
// A storage tier is named in the catalog (devices/storage.yaml) and is nothing more there: what
// moving bytes to or from it costs is a rate and a floor, which are fitted or assumed and so
// belong in the registry, scoped to the hardware they describe. The tier rides in the entry's
// name, as a data type or a backend does elsewhere, because a scope has no tier dimension.
//
// There are three per tier, the three terms of the kernel's TierTime, which is max(floor,
// bytes / rate) per direction:
//
//	offload_tier_<tier>_read_rate     bytes per microsecond, tier to GPU
//	offload_tier_<tier>_write_rate    bytes per microsecond, GPU to tier
//	offload_tier_<tier>_base_latency  microseconds per transfer
//
// A rate in bytes per microsecond is numerically a rate in MB/s.

// OffloadTierPrefix starts the name of every offload-tier coefficient.
const OffloadTierPrefix = "offload_tier_"

// OffloadTierReadRate names the rate at which a tier delivers bytes to the GPU.
func OffloadTierReadRate(tier string) string { return OffloadTierPrefix + tier + "_read_rate" }

// OffloadTierWriteRate names the rate at which the GPU writes bytes to a tier.
func OffloadTierWriteRate(tier string) string { return OffloadTierPrefix + tier + "_write_rate" }

// OffloadTierBaseLatency names the fixed cost of one transfer, in microseconds.
func OffloadTierBaseLatency(tier string) string { return OffloadTierPrefix + tier + "_base_latency" }
