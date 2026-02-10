package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// Model represents a model that can be deployed with various configurations
type Model struct {
	ModelID    ModelID
	Parameters int64 // Total model parameters
	Configs    []DeploymentConfig
}

// DeploymentConfig represents a specific deployment configuration for a model
type DeploymentConfig struct {
	ConfigID       ConfigID
	ModelID        ModelID
	Architecture   ArchitectureType // MONOLITHIC only in Phase 1

	// Configuration components
	ModelConfig    *HFModelConfig
	EngineConfig   *VLLMEngineConfig
	HardwareConfig *sim.HardwareCalib // GPU specs (reuse existing type)

	// Performance model coefficients
	AlphaCoeffs []float64 // Queueing delay model coefficients
	BetaCoeffs  []float64 // Step time model coefficients

	// Replica pool (monolithic only in Phase 1)
	ReplicaPool *ReplicaPool
}

// ReplicaPool manages a pool of instances for a deployment configuration
type ReplicaPool struct {
	PoolID      string
	PoolType    PoolType
	Instances   []*InstanceSimulator
	MinReplicas int
	MaxReplicas int
}

// NewReplicaPool creates a new replica pool with the given bounds
func NewReplicaPool(poolID string, poolType PoolType, minReplicas, maxReplicas int) (*ReplicaPool, error) {
	if minReplicas < 0 {
		return nil, fmt.Errorf("MinReplicas must be >= 0, got %d", minReplicas)
	}
	if maxReplicas < minReplicas {
		return nil, fmt.Errorf("MaxReplicas (%d) must be >= MinReplicas (%d)", maxReplicas, minReplicas)
	}

	return &ReplicaPool{
		PoolID:      poolID,
		PoolType:    poolType,
		Instances:   make([]*InstanceSimulator, 0, maxReplicas),
		MinReplicas: minReplicas,
		MaxReplicas: maxReplicas,
	}, nil
}

// AddInstance adds a new instance to the pool
// Returns error if pool is at MaxReplicas
func (p *ReplicaPool) AddInstance(instance *InstanceSimulator) error {
	if len(p.Instances) >= p.MaxReplicas {
		return fmt.Errorf("cannot add instance: pool at MaxReplicas (%d)", p.MaxReplicas)
	}

	p.Instances = append(p.Instances, instance)
	return nil
}

// RemoveInstance removes an instance from the pool by ID
// Returns error if pool is at MinReplicas or instance not found
func (p *ReplicaPool) RemoveInstance(id InstanceID) error {
	if len(p.Instances) <= p.MinReplicas {
		return fmt.Errorf("cannot remove instance: pool at MinReplicas (%d)", p.MinReplicas)
	}

	for i, inst := range p.Instances {
		if inst.ID == id {
			// Remove by replacing with last element and truncating
			p.Instances[i] = p.Instances[len(p.Instances)-1]
			p.Instances = p.Instances[:len(p.Instances)-1]
			return nil
		}
	}

	return fmt.Errorf("instance %s not found in pool", id)
}

// GetInstance retrieves an instance by ID
func (p *ReplicaPool) GetInstance(id InstanceID) *InstanceSimulator {
	for _, inst := range p.Instances {
		if inst.ID == id {
			return inst
		}
	}
	return nil
}

// Len returns the current number of instances in the pool
func (p *ReplicaPool) Len() int {
	return len(p.Instances)
}
