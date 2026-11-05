package latencykernel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"

	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// fabricEfficiencyName is the registry entry that corrects a fabric's nominal line rate to
// what a transfer over it achieves: measured payload throughput over the catalog's
// InterNodeBwGBps, dimensionless, scoped by the fabric's catalog identity (blis-schemas
// coefficient.Scope.Fabric).
//
// The catalog states only the nominal figure, a line rate divided by eight, and says why:
// the effective fraction is a per-fabric measurement and belongs in the registry. Two
// clusters of comparable silicon on different networks -- an H100 cluster over RoCE and an
// H200 cluster over InfiniBand -- differ along exactly this axis, so without it both price
// every byte that leaves a node at their datasheet rate.
//
// Where it applies: every term that prices bytes over the inter-node fabric. That is the
// NIC rate a P/D handoff and a pipeline-stage boundary cross (nicBytesPerSecond), and the
// intra-to-inter ratio a cross-node collective's bytes are scaled by (resolve.Fabric.Ratio).
// An NVLink domain that spans a rack is the chip's, and a correction to it is scoped by
// hardware, not by this entry.
const fabricEfficiencyName = "fabric_bandwidth_efficiency"

// resolveFabricEfficiency reads the fabric's efficiency. It returns 1 and stated false for a
// cluster with no fabric or a registry that carries none for it, which the caller records as
// an assumption. A value outside (0, 1] is refused: above one would price a link faster than
// its line rate, and zero or below would make a transfer free or infinite.
//
// Only an entry scoped to a fabric is read. An entry of this name scoped only by hardware or
// model would apply over every fabric, which is the nominal-rate error this term corrects;
// it is skipped rather than allowed to hide the fabric's own entry, wherever the two sit in
// the scenario's sets (resolve.Coefficients.EntryScopedToFabric).
func resolveFabricEfficiency(c *resolve.Coefficients, f *hardware.Fabric) (float64, bool, error) {
	if f == nil {
		return 1, true, nil
	}
	r, ok := c.EntryScopedToFabric(fabricEfficiencyName)
	if !ok {
		return 1, false, nil
	}
	e := r.Entry.Value
	if !(e > 0 && e <= 1) {
		return 0, false, fmt.Errorf("%s is %v for fabric %s, outside (0, 1]: a fabric "+
			"cannot carry more than its line rate", fabricEfficiencyName, e, f.Name)
	}
	return e, true, nil
}
