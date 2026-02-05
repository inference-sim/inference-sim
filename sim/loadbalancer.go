package sim

import (
	"github.com/sirupsen/logrus"
)

// LoadBalancer defines the interface for routing requests to replicas
type LoadBalancer interface {
	// GetReplica selects which replica should handle the incoming request
	// Parameters:
	//   e: the arrival event containing request information
	//   replicas: current state of all replicas (for state-aware routing)
	// Returns: index of the selected replica
	GetReplica(e *ArrivalEvent, replicas []Replica) int
}

// NewLoadBalancer creates a load balancer of the specified type
func NewLoadBalancer(lbType string, numReplicas int, seed int64) LoadBalancer {
	switch lbType {
	case "random":
		return NewRandomLoadBalancer(numReplicas, seed)
	default:
		logrus.Panicf("unknown load balancer type: %s", lbType)
		return nil
	}
}

// GetAvailableLoadBalancers returns list of supported load balancer types
func GetAvailableLoadBalancers() []string {
	return []string{"random"}
}
