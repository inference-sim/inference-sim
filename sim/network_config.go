package sim

import (
	"fmt"
	"math"
)

// NetworkConfig describes the inter-node interconnect for the trained-physics
// latency model (#1530). It is the topology + fabric input the model uses to
// charge cross-node collective traffic (multi-node TP all-reduce, DEP expert
// all-to-all) at an inter-node bandwidth/latency instead of the on-package
// HBM/NVLink bandwidth.
//
// The zero value is inert: with GPUsPerNode == 0 the model treats every collective
// as intra-node and StepTime is byte-identical to a pre-feature build (INV-6,
// INV-BC-DP1). The network term contributes exactly zero unless a collective's
// participant group exceeds GPUsPerNode — i.e. a real node boundary is crossed.
//
// Topology is expressed as an explicit config input (mirroring how --tp/--dp are
// modeling inputs), NOT derived from node_pools placement. This keeps the signal
// identical across `blis run` and `blis replay` (both build it from the same CLI
// flags), so run/replay parity (INV-13) holds by construction without any TraceV2
// round-trip. Deriving GPUsPerNode from real multi-node placement is the #1529
// follow-up that consumes this same seam.
type NetworkConfig struct {
	// GPUsPerNode is the number of GPUs sharing the fast intra-node fabric (NVLink).
	// A parallel collective whose group size exceeds this spans node boundaries.
	// 0 = unset ⇒ the network model is inert (single-node, no cross-node cost).
	GPUsPerNode int

	// InterNodeBandwidthGBps is the effective cross-node link bandwidth in GB/s
	// (InfiniBand / RoCE). Used only when a collective crosses a node boundary.
	// Interpreted as an effective achievable rate (like --pd-transfer-bandwidth),
	// not a theoretical peak. Must be > 0 when GPUsPerNode > 0 (R11 divisor guard).
	InterNodeBandwidthGBps float64

	// InterNodeLatencyMs is the fixed per-collective cross-node base latency in ms
	// (switch/hop latency). 0 = no fixed latency. Non-negative.
	InterNodeLatencyMs float64
}

// NewNetworkConfig creates a NetworkConfig with all fields explicitly set. This is
// the canonical constructor — all construction sites must use it (R4).
func NewNetworkConfig(gpusPerNode int, interNodeBandwidthGBps, interNodeLatencyMs float64) NetworkConfig {
	return NetworkConfig{
		GPUsPerNode:            gpusPerNode,
		InterNodeBandwidthGBps: interNodeBandwidthGBps,
		InterNodeLatencyMs:     interNodeLatencyMs,
	}
}

// IsActive reports whether the network model contributes any cost. It is active
// exactly when GPUsPerNode > 0; when inactive the latency model is byte-identical
// to a pre-feature build (INV-6).
func (c NetworkConfig) IsActive() bool {
	return c.GPUsPerNode > 0
}

// Validate checks the NetworkConfig for internal consistency. It is a pure query —
// the library boundary returns an error and the caller decides fatality (cmd/ →
// logrus.Fatalf; sim/ factory → error). The zero value is valid and inert (INV-6).
//
// Guards (R1/R3/R11):
//   - GPUsPerNode must be non-negative.
//   - Bandwidth and latency must be finite and non-negative.
//   - An active config (GPUsPerNode > 0) requires a positive bandwidth (the
//     cross-node cost divides by it).
//   - Fabric knobs set without GPUsPerNode is a misconfiguration — they would have
//     no effect, so surface it loudly rather than silently ignore (R1).
func (c NetworkConfig) Validate() error {
	if c.GPUsPerNode < 0 {
		return fmt.Errorf("NetworkConfig: GPUsPerNode must be >= 0, got %d", c.GPUsPerNode)
	}
	if math.IsNaN(c.InterNodeBandwidthGBps) || math.IsInf(c.InterNodeBandwidthGBps, 0) || c.InterNodeBandwidthGBps < 0 {
		return fmt.Errorf("NetworkConfig: InterNodeBandwidthGBps must be a finite non-negative number, got %v", c.InterNodeBandwidthGBps)
	}
	if math.IsNaN(c.InterNodeLatencyMs) || math.IsInf(c.InterNodeLatencyMs, 0) || c.InterNodeLatencyMs < 0 {
		return fmt.Errorf("NetworkConfig: InterNodeLatencyMs must be a finite non-negative number, got %v", c.InterNodeLatencyMs)
	}
	if c.GPUsPerNode > 0 && c.InterNodeBandwidthGBps <= 0 {
		return fmt.Errorf("NetworkConfig: InterNodeBandwidthGBps must be > 0 when GPUsPerNode is set (got GPUsPerNode=%d, bandwidth=%v)", c.GPUsPerNode, c.InterNodeBandwidthGBps)
	}
	if c.GPUsPerNode == 0 && (c.InterNodeBandwidthGBps > 0 || c.InterNodeLatencyMs > 0) {
		return fmt.Errorf("NetworkConfig: inter-node bandwidth/latency have no effect without GPUsPerNode > 0 (got bandwidth=%v, latency=%v ms); set --gpus-per-node", c.InterNodeBandwidthGBps, c.InterNodeLatencyMs)
	}
	return nil
}
