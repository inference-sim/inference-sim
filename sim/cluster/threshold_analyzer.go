package cluster

import (
	"fmt"
	"math"
	"sort"
)

// The threshold analyzer (inference-sim#2063): a target-value rule over one per-replica signal, the
// shape of llm-d's supported autoscaling path (KEDA reading a metric, HPA computing
// desired = ceil(sum(metric) / target)). It fits the Analyzer contract with no new engine:
//
//	TotalSupply      = target x ready replicas
//	TotalDemand      = sum of the signal over the ready replicas
//	RequiredCapacity = max(0, TotalDemand - (TotalSupply + target x replicas still loading))
//
// UnlimitedEngine.scaleUpN then adds ceil(Required / target) = ceil(sum / target) - replicas, the HPA
// desired-replicas step. Scale-down is the symmetric floor, and never takes the model below
// MinReplicas. It is not the Kubernetes HPA: tolerance, behaviour policies and the metrics
// pipeline's lags are not modelled.

// Analyzer kinds a deployment can select.
const (
	AnalyzerKindV2Saturation = "v2-saturation" // the default
	AnalyzerKindThreshold    = "threshold"
)

// Signals the threshold analyzer can read from ReplicaMetrics.
const (
	SignalInFlight      = "in_flight"
	SignalQueueDepth    = "queue_depth"
	SignalKVUtilization = "kv_utilization"
)

// AnalyzerSelection chooses the analyzer of the model autoscaler. The zero value is v2-saturation.
type AnalyzerSelection struct {
	Kind   string  `yaml:"kind"`
	Signal string  `yaml:"signal"`
	Target float64 `yaml:"target"`
}

// Validate refuses a selection that cannot run.
func (s AnalyzerSelection) Validate() error {
	switch s.Kind {
	case "", AnalyzerKindV2Saturation:
		if s.Signal != "" || s.Target != 0 {
			return fmt.Errorf("autoscaler analyzer: signal and target belong to kind %q; kind is %q", AnalyzerKindThreshold, orDefault(s.Kind))
		}
		return nil
	case AnalyzerKindThreshold:
	default:
		return fmt.Errorf("autoscaler analyzer kind %q is not an analyzer; they are %s and %s", s.Kind, AnalyzerKindV2Saturation, AnalyzerKindThreshold)
	}
	switch s.Signal {
	case SignalInFlight, SignalQueueDepth, SignalKVUtilization:
	default:
		return fmt.Errorf("autoscaler analyzer signal %q is not a signal; they are %s, %s and %s",
			s.Signal, SignalInFlight, SignalQueueDepth, SignalKVUtilization)
	}
	if math.IsNaN(s.Target) || math.IsInf(s.Target, 0) || s.Target <= 0 {
		return fmt.Errorf("autoscaler analyzer target must be a finite number > 0, got %v", s.Target)
	}
	return nil
}

func orDefault(kind string) string {
	if kind == "" {
		return AnalyzerKindV2Saturation
	}
	return kind
}

// ThresholdAnalyzer implements Analyzer.
type ThresholdAnalyzer struct {
	signal      string
	target      float64
	minReplicas int
}

// NewThresholdAnalyzer constructs the analyzer; it panics on an invalid selection (R4).
func NewThresholdAnalyzer(sel AnalyzerSelection, minReplicas int) *ThresholdAnalyzer {
	if sel.Kind != AnalyzerKindThreshold {
		panic(fmt.Sprintf("NewThresholdAnalyzer: kind %q", sel.Kind))
	}
	if err := sel.Validate(); err != nil {
		panic("NewThresholdAnalyzer: " + err.Error())
	}
	return &ThresholdAnalyzer{signal: sel.Signal, target: sel.Target, minReplicas: max(minReplicas, 1)}
}

// Name returns the analyzer name for observability.
func (a *ThresholdAnalyzer) Name() string { return AnalyzerKindThreshold }

func (a *ThresholdAnalyzer) value(r ReplicaMetrics) float64 {
	switch a.signal {
	case SignalQueueDepth:
		return float64(r.QueueDepth)
	case SignalKVUtilization:
		return r.KVUtilization
	default:
		return float64(r.InFlightCount)
	}
}

// Analyze computes the model's supply, demand and the scale signals.
func (a *ThresholdAnalyzer) Analyze(ms ModelSignals) AnalyzerResult {
	res := AnalyzerResult{ModelID: ms.ModelID}
	if len(ms.Replicas) == 0 {
		return res
	}
	type agg struct {
		demand float64
		n      int
		cost   float64
	}
	variants := map[VariantSpec]*agg{}
	for _, r := range ms.Replicas {
		v, ok := variants[r.Variant]
		if !ok {
			v = &agg{cost: r.CostPerHour}
			variants[r.Variant] = v
		}
		v.demand += a.value(r)
		v.n++
	}
	vcs := make([]VariantCapacity, 0, len(variants))
	for variant, v := range variants {
		vcs = append(vcs, VariantCapacity{Variant: variant, Supply: a.target * float64(v.n), Demand: v.demand,
			ReplicaCount: v.n, CostPerReplica: v.cost})
	}
	sort.Slice(vcs, func(i, j int) bool { // cost ascending, then name (R2)
		if vcs[i].CostPerReplica != vcs[j].CostPerReplica {
			return vcs[i].CostPerReplica < vcs[j].CostPerReplica
		}
		if vcs[i].Variant.GPUType != vcs[j].Variant.GPUType {
			return vcs[i].Variant.GPUType < vcs[j].Variant.GPUType
		}
		return vcs[i].Variant.TPDegree < vcs[j].Variant.TPDegree
	})
	res.VariantCapacities = vcs
	ready := 0
	for _, vc := range vcs { // summed in the sorted order (R2)
		res.TotalSupply += vc.Supply
		res.TotalDemand += vc.Demand
		ready += vc.ReplicaCount
	}
	if res.TotalSupply > 0 {
		res.Utilization = res.TotalDemand / res.TotalSupply
	}
	// Replicas still loading count as supply for the scale-up signal only, as the saturation
	// analyzer's pending supply does (#1109): without it every tick of a slow load asks again.
	pending := a.target * float64(ms.PendingReplicaCount)
	if need := res.TotalDemand - (res.TotalSupply + pending); need > 0 {
		res.RequiredCapacity = need
		return res
	}
	// Scale-down: the replicas the demand does not need, at the target, above the floor.
	desired := max(int(math.Ceil(res.TotalDemand/a.target)), a.minReplicas)
	if safe := ready - desired; safe > 0 {
		res.SafeRemovals = safe
		res.SpareCapacity = res.TotalSupply - res.TotalDemand
	}
	return res
}
