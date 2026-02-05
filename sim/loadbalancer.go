package sim

import "math/rand"

type LoadBalancer interface {
	GetReplica(e *ArrivalEvent, replicas []Replica) int
}

type RandomLoadBalancer struct {
	NumReplicas int
	rand        *rand.Rand
}

func (lb *RandomLoadBalancer) GetReplica(e *ArrivalEvent, replicas []Replica) int {
	return lb.rand.Intn(lb.NumReplicas)
}

func NewRandomLoadBalancer(numReplicas int, seed int64) *RandomLoadBalancer {
	lb := RandomLoadBalancer{
		NumReplicas: numReplicas,
		rand:        rand.New(rand.NewSource(seed)),
	}
	return &lb
}
