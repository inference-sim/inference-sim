package sim

import "math/rand"

// RandomLoadBalancer routes requests randomly across replicas
type RandomLoadBalancer struct {
	NumReplicas int
	rand        *rand.Rand
}

// GetReplica selects a random replica index
func (lb *RandomLoadBalancer) GetReplica(e *ArrivalEvent, replicas []Replica) int {
	return lb.rand.Intn(lb.NumReplicas)
}

// NewRandomLoadBalancer creates a random load balancer
func NewRandomLoadBalancer(numReplicas int, seed int64) *RandomLoadBalancer {
	lb := RandomLoadBalancer{
		NumReplicas: numReplicas,
		rand:        rand.New(rand.NewSource(seed)),
	}
	return &lb
}
