package plan

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

// Memory is what a plan states about where its bytes live (SUB-14). GPU figures are per rank.
// Host figures are per node: a shared figure counts once per node however many engines run on
// it, and an unshared one counts once per rank on the node.
type Memory struct {
	// WeightsGPU is the resident weights on one rank, and BuffersGPU the buffers and workspaces
	// the engine chose on one rank. The cost model adds activations, graph pools and the
	// libraries' own memory, which are its to size (design §6.3).
	WeightsGPU units.PerRank[units.Bytes]
	BuffersGPU units.PerRank[units.Bytes]
	// Host is the host memory the plan uses on one node.
	Host units.PerNode[units.Bytes]
}

// MemoryOf sums a plan's stated figures. ranksOnNode is how many of the pool's ranks share a
// node; it multiplies an unshared host figure and leaves a shared one alone.
func MemoryOf(p PoolPlan, ranksOnNode int) (Memory, error) {
	if ranksOnNode < 1 {
		return Memory{}, fmt.Errorf("plan: %d ranks on a node", ranksOnNode)
	}
	var m Memory
	hostPerRank := units.Bytes(0)
	hostShared := units.Bytes(0)
	for i, w := range p.Weights {
		b, err := w.Bytes()
		if err != nil {
			return Memory{}, fmt.Errorf("plan: Weights[%d]: %w", i, err)
		}
		switch {
		case w.Where == ResidentHost && w.Shared:
			hostShared += b.Value()
		case w.Where == ResidentHost:
			hostPerRank += b.Value()
		default:
			m.WeightsGPU = m.WeightsGPU.Add(b)
		}
	}
	for i, b := range p.Buffers {
		if !b.Basis.Valid() {
			return Memory{}, fmt.Errorf("plan: Buffers[%d] %q has basis %q", i, b.Name, b.Basis)
		}
		switch {
		case b.Host && b.Basis == BasisNode:
			hostShared += b.Bytes
		case b.Host && b.Basis == BasisRank:
			hostPerRank += b.Bytes
		case b.Host:
			return Memory{}, fmt.Errorf("plan: Buffers[%d] %q is a host buffer for the whole group; state it per rank or per node", i, b.Name)
		case b.Basis == BasisRank:
			m.BuffersGPU = m.BuffersGPU.Add(units.Rank(b.Bytes))
		default:
			return Memory{}, fmt.Errorf("plan: Buffers[%d] %q is a GPU buffer on basis %q; GPU buffers are per rank", i, b.Name, b.Basis)
		}
	}
	// KV tiers are counted by HostTierBytes: whether a tier device is host memory is the
	// catalog's to say.
	perRank, err := units.Rank(hostPerRank).Across(ranksOnNode)
	if err != nil {
		return Memory{}, err
	}
	m.Host = units.Node(perRank.Value() + hostShared)
	return m, nil
}

// HostTierBytes returns the bytes of the plan's tiers whose device the isHost predicate says is
// host memory, on one node: a shared tier counts once, an unshared one per rank.
func HostTierBytes(p PoolPlan, ranksOnNode int, isHost func(device string) bool) (units.PerNode[units.Bytes], error) {
	if ranksOnNode < 1 {
		return units.PerNode[units.Bytes]{}, fmt.Errorf("plan: %d ranks on a node", ranksOnNode)
	}
	var total units.Bytes
	for _, t := range p.Tiers {
		if !isHost(t.Device) {
			continue
		}
		if t.Shared {
			total += t.Bytes.Value()
		} else {
			total += t.Bytes.Value() * units.Bytes(ranksOnNode)
		}
	}
	return units.Node(total), nil
}

// CheckHostFits refuses a node whose host figures exceed its host memory (SUB-14): the
// weights an engine keeps in host memory, its host KV tier and its pinned buffers, summed for
// one node. hostMemory is the catalog's host memory per node.
func CheckHostFits(used, hostMemory units.PerNode[units.Bytes]) error {
	if hostMemory.Value() <= 0 {
		return fmt.Errorf("plan: the node's host memory is %d bytes; it must be stated to check the plan against it", hostMemory.Value())
	}
	if used.Value() > hostMemory.Value() {
		return fmt.Errorf("plan: host figures of %d bytes on a node exceed its %d bytes of host memory", used.Value(), hostMemory.Value())
	}
	return nil
}
