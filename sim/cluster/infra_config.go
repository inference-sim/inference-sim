// infra_config.go defines configuration types for node pools and instance lifecycle.
// Phase 1A: NodePoolConfig, InstanceLifecycleConfig, DrainPolicy, DelaySpec.
package cluster

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// subsystem names for PartitionedRNG (INV-6: determinism requires named subsystems).
const (
	subsystemNodeProvisioning = "node-provisioning"
	subsystemInstanceLoading  = "instance-loading"
)

// DelaySpec parameterizes a duration distribution for provisioning/loading delays.
// Delays are expressed in seconds; Sample() converts to microsecond ticks (int64).
type DelaySpec struct {
	Mean   float64 `yaml:"mean"`   // seconds ≥0; mean delay
	Stddev float64 `yaml:"stddev"` // seconds ≥0; 0 = constant (no variance)
}

// Sample draws a delay and returns it as microsecond ticks (int64).
// Uses Gaussian sampling when Stddev > 0; returns Mean as constant otherwise.
// Result is clamped to ≥0 (negative samples return 0).
func (d DelaySpec) Sample(rng *rand.Rand) int64 {
	var secs float64
	if d.Stddev <= 0 {
		secs = d.Mean
	} else {
		secs = rng.NormFloat64()*d.Stddev + d.Mean
		if secs < 0 {
			secs = 0
		}
	}
	// Defense-in-depth: clamp NaN/Inf to zero (C2). Validation should catch this
	// upstream, but corrupt config or floating-point edge cases could slip through.
	if math.IsNaN(secs) || math.IsInf(secs, 0) {
		secs = 0
	}
	return int64(secs * 1e6) // convert seconds → microseconds
}

// IsZero returns true if the delay is always zero (no-op).
func (d DelaySpec) IsZero() bool {
	return d.Mean == 0 && d.Stddev == 0
}

// NodePoolConfig configures a pool of homogeneous nodes.
// All fields are strictly validated by IsValid(). (R3, R10)
type NodePoolConfig struct {
	Name              string    `yaml:"name"`
	GPUType           string    `yaml:"gpu_type"`
	GPUsPerNode       int       `yaml:"gpus_per_node"`
	GPUMemoryGiB      float64   `yaml:"gpu_memory_gib"`
	InitialNodes      int       `yaml:"initial_nodes"`
	MinNodes          int       `yaml:"min_nodes"`
	MaxNodes          int       `yaml:"max_nodes"`
	ProvisioningDelay DelaySpec `yaml:"provisioning_delay"`
}

// IsValid validates all fields in the NodePoolConfig.
// Returns a descriptive error for the first violation found.
func (c *NodePoolConfig) IsValid() error {
	if c.Name == "" {
		return errors.New("node pool name must not be empty")
	}
	if c.GPUType == "" {
		return fmt.Errorf("node pool %q: gpu_type must not be empty", c.Name)
	}
	if c.GPUsPerNode < 1 {
		return fmt.Errorf("node pool %q: gpus_per_node must be ≥1, got %d", c.Name, c.GPUsPerNode)
	}
	if math.IsNaN(c.GPUMemoryGiB) || math.IsInf(c.GPUMemoryGiB, 0) {
		return fmt.Errorf("node pool %q: gpu_memory_gib must not be NaN or Inf", c.Name)
	}
	if c.GPUMemoryGiB <= 0 {
		return fmt.Errorf("node pool %q: gpu_memory_gib must be >0, got %g", c.Name, c.GPUMemoryGiB)
	}
	if c.InitialNodes < 0 {
		return fmt.Errorf("node pool %q: initial_nodes must be ≥0, got %d", c.Name, c.InitialNodes)
	}
	if c.MinNodes < 0 {
		return fmt.Errorf("node pool %q: min_nodes must be ≥0, got %d", c.Name, c.MinNodes)
	}
	if c.MaxNodes < c.InitialNodes {
		return fmt.Errorf("node pool %q: max_nodes (%d) must be ≥ initial_nodes (%d)", c.Name, c.MaxNodes, c.InitialNodes)
	}
	if c.MinNodes > c.MaxNodes {
		return fmt.Errorf("node pool %q: min_nodes (%d) must be ≤ max_nodes (%d)", c.Name, c.MinNodes, c.MaxNodes)
	}
	if math.IsNaN(c.ProvisioningDelay.Mean) || math.IsInf(c.ProvisioningDelay.Mean, 0) {
		return fmt.Errorf("node pool %q: provisioning_delay.mean must not be NaN or Inf", c.Name)
	}
	if c.ProvisioningDelay.Mean < 0 {
		return fmt.Errorf("node pool %q: provisioning_delay.mean must be ≥0", c.Name)
	}
	if math.IsNaN(c.ProvisioningDelay.Stddev) || math.IsInf(c.ProvisioningDelay.Stddev, 0) {
		return fmt.Errorf("node pool %q: provisioning_delay.stddev must not be NaN or Inf", c.Name)
	}
	if c.ProvisioningDelay.Stddev < 0 {
		return fmt.Errorf("node pool %q: provisioning_delay.stddev must be ≥0", c.Name)
	}
	return nil
}

// DrainPolicyName represents the drain behavior on instance termination.
type DrainPolicyName string

const (
	DrainPolicyImmediate DrainPolicyName = "IMMEDIATE"
	DrainPolicyWait      DrainPolicyName = "WAIT"
	DrainPolicyRedirect  DrainPolicyName = "REDIRECT"
)

// validDrainPolicies is the unexported validation map (R8).
var validDrainPolicies = map[DrainPolicyName]struct{}{
	DrainPolicyImmediate: {},
	DrainPolicyWait:      {},
	DrainPolicyRedirect:  {},
}

// IsValidDrainPolicy returns true if name is a known drain policy value.
func IsValidDrainPolicy(name string) bool {
	_, ok := validDrainPolicies[DrainPolicyName(name)]
	return ok
}

// InstanceLifecycleConfig configures per-instance timing for lifecycle transitions.
// Zero values are safe defaults: no loading delay, no warm-up, WAIT drain policy.
type InstanceLifecycleConfig struct {
	LoadingDelay       DelaySpec `yaml:"loading_delay"`
	WarmUpRequestCount int       `yaml:"warm_up_request_count"`
	WarmUpTTFTFactor   float64   `yaml:"warm_up_ttft_factor"`
	DrainPolicy        string    `yaml:"drain_policy"` // "IMMEDIATE", "WAIT", or "REDIRECT"
}

// IsValid validates all fields in the InstanceLifecycleConfig.
func (c *InstanceLifecycleConfig) IsValid() error {
	if c.WarmUpRequestCount < 0 {
		return fmt.Errorf("warm_up_request_count must be ≥0, got %d", c.WarmUpRequestCount)
	}
	factor := c.WarmUpTTFTFactor
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return fmt.Errorf("warm_up_ttft_factor must not be NaN or Inf")
	}
	if factor != 0 && factor < 1.0 {
		return fmt.Errorf("warm_up_ttft_factor must be ≥1.0, got %g", factor)
	}
	if math.IsNaN(c.LoadingDelay.Mean) || math.IsInf(c.LoadingDelay.Mean, 0) {
		return errors.New("loading_delay.mean must not be NaN or Inf")
	}
	if c.LoadingDelay.Mean < 0 {
		return errors.New("loading_delay.mean must be ≥0")
	}
	if math.IsNaN(c.LoadingDelay.Stddev) || math.IsInf(c.LoadingDelay.Stddev, 0) {
		return errors.New("loading_delay.stddev must not be NaN or Inf")
	}
	if c.LoadingDelay.Stddev < 0 {
		return errors.New("loading_delay.stddev must be ≥0")
	}
	if c.DrainPolicy != "" && !IsValidDrainPolicy(c.DrainPolicy) {
		return fmt.Errorf("unknown drain_policy %q (valid: IMMEDIATE, WAIT, REDIRECT)", c.DrainPolicy)
	}
	return nil
}

// effectiveWarmUpFactor returns the warm-up TTFT factor, defaulting to 1.0.
func (c *InstanceLifecycleConfig) effectiveWarmUpFactor() float64 {
	if c.WarmUpTTFTFactor == 0 {
		return 1.0
	}
	return c.WarmUpTTFTFactor
}
