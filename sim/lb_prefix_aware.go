package sim

import (
	"math/rand"
)

// PrefixAwareLoadBalancer routes requests to the replica with the most cached prefix blocks
type PrefixAwareLoadBalancer struct {
	NumReplicas int
	rand        *rand.Rand
}

// NewPrefixAwareLoadBalancer creates a new prefix-aware load balancer
func NewPrefixAwareLoadBalancer(numReplicas int, seed int64) *PrefixAwareLoadBalancer {
	return &PrefixAwareLoadBalancer{
		NumReplicas: numReplicas,
		rand:        rand.New(rand.NewSource(seed)),
	}
}

// GetReplica selects the replica with the most cached prefix blocks
func (lb *PrefixAwareLoadBalancer) GetReplica(e *ArrivalEvent, replicas []Replica) int {
	maxScore := -1
	var candidates []int

	// Score each replica by counting cached prefix blocks
	for i := range replicas {
		cachedBlocks := replicas[i].KVCache.GetCachedBlocks(e.Request.InputTokens)
		score := len(cachedBlocks)

		if score > maxScore {
			maxScore = score
			candidates = []int{i}
		} else if score == maxScore {
			candidates = append(candidates, i)
		}
	}

	// Random tiebreaker among replicas with equal max score
	if len(candidates) > 0 {
		return candidates[lb.rand.Intn(len(candidates))]
	}

	// Fallback: random selection (shouldn't happen)
	return lb.rand.Intn(lb.NumReplicas)
}
