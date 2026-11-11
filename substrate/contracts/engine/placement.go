package engine

import (
	"errors"
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// Placeable is implemented by an engine that can be told where its instance runs. The host owns
// placement: it knows which node and rack of the cluster each instance of a pool sits on, and the
// engine cannot learn it any other way. An engine that states the endpoints of its transfers
// (plan.Transfer.From and To, design §6.2) states its own end from the endpoint the host gave it,
// so two instances of one pool on different nodes report different endpoints, and the cost model
// prices the links each uses (SUB-17).
//
// at is the instance's node and rack in the pool's own numbering, the numbering Request.Peer
// uses for the far end. PlaceAt returns an error for a placement the engine cannot honour; it does
// not choose another. It is a separate, optional interface so that an engine that starts no
// transfer, or serves one node, needs no placement; Place is the host's one call and decides when
// the absence of the interface is a refusal.
type Placeable interface {
	PlaceAt(at plan.Endpoint) error
}

// ErrPlacementUnsupported is wrapped by the error Place returns when an engine must be placed and
// does not implement Placeable: its transfers would report the pool's first node whatever node
// the instance runs on, and the links priced would be wrong without the run saying so.
var ErrPlacementUnsupported = errors.New("engine: placement unsupported")

// ErrPlacementRejected is wrapped by the error Place returns when the engine's PlaceAt refuses
// the placement, or the placement is not a node and rack of the pool.
var ErrPlacementRejected = errors.New("engine: placement rejected")

// Place tells the engine of one instance where it runs. poolNodes is the number of nodes of the
// pool the instance belongs to. Placement is required when the pool spans more than one node, or
// the instance is not at the first node and rack: then an engine that is not Placeable is
// refused with ErrPlacementUnsupported, and one whose PlaceAt returns an error with
// ErrPlacementRejected. An instance of a one-node pool at node 0, rack 0 needs no placement; the
// engine is told when it can be. A host calls Place for every instance of a disaggregated pool,
// and starts no run on an error.
func Place(e Engine, at plan.Endpoint, poolNodes int) error {
	if at.Node < 0 || at.Rack < 0 || (poolNodes > 0 && at.Node >= poolNodes) {
		return &PlacementError{At: at, PoolNodes: poolNodes, Err: fmt.Errorf("%w: node %d rack %d is not a place in a pool of %d node(s)", ErrPlacementRejected, at.Node, at.Rack, poolNodes)}
	}
	required := poolNodes > 1 || at != (plan.Endpoint{})
	p, ok := e.(Placeable)
	if !ok {
		if !required {
			return nil
		}
		return &PlacementError{At: at, PoolNodes: poolNodes, Err: fmt.Errorf("%w: the engine does not implement engine.Placeable, so its transfers would name the pool's first node for an instance at node %d rack %d of a pool of %d node(s)", ErrPlacementUnsupported, at.Node, at.Rack, poolNodes)}
	}
	if err := p.PlaceAt(at); err != nil {
		return &PlacementError{At: at, PoolNodes: poolNodes, Err: fmt.Errorf("%w: %v", ErrPlacementRejected, err)}
	}
	return nil
}

// PlacementError is the refusal Place returns: where the instance was to run, in a pool of how
// many nodes, and which of ErrPlacementUnsupported and ErrPlacementRejected it is (errors.Is).
type PlacementError struct {
	At        plan.Endpoint
	PoolNodes int
	Err       error
}

func (e *PlacementError) Error() string { return e.Err.Error() }
func (e *PlacementError) Unwrap() error { return e.Err }
