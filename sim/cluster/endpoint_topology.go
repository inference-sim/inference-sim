package cluster

// EndpointTopology is the location a kernel-backed scenario may truthfully
// advertise to the router. NodeCount is zero when placement is unknown; a
// worker occupying more than one node has no single hostname.
type EndpointTopology struct {
	Hostname  string
	Rack      string
	Zone      string
	Region    string
	NodeCount int
	// MixedSources means the scenario's kernel placement and NodePool placement
	// coexist. Their coordinate systems are independent, so topology-aware
	// routing cannot safely assume they agree with KV-transfer pricing.
	MixedSources bool
}

// EndpointTopologyResolver returns the immutable scenario placement captured
// for one worker. A nil resolver means the scenario provides no placement.
type EndpointTopologyResolver func(InstanceID) EndpointTopology

// StaticEndpointTopology snapshots a scenario's per-worker placements so
// callers cannot mutate the locations after cluster construction.
func StaticEndpointTopology(entries map[InstanceID]EndpointTopology) EndpointTopologyResolver {
	if len(entries) == 0 {
		return nil
	}
	snapshot := make(map[InstanceID]EndpointTopology, len(entries))
	for id, t := range entries {
		snapshot[id] = t
	}
	return func(id InstanceID) EndpointTopology { return snapshot[id] }
}

// EndpointTopology returns the location captured when this worker was placed.
func (i *InstanceSimulator) EndpointTopology() EndpointTopology { return i.endpointTopology }
