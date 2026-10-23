package llmd

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"sort"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	sigsyaml "sigs.k8s.io/yaml"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

const rackLabel = "topology.kubernetes.io/rack"

// nodeLocations reads explicit simulated Kubernetes Node labels. The scenario
// determines which node a worker occupies; a Node manifest can additionally
// state its zone and region, which the scenario and fabric do not define.
func nodeLocations(nodes []*corev1.Node) (map[string]cluster.EndpointTopology, error) {
	if len(nodes) == 0 {
		return nil, nil
	}
	out := make(map[string]cluster.EndpointTopology, len(nodes))
	for _, node := range nodes {
		name := node.Name
		if name == "" {
			return nil, fmt.Errorf("a simulated Node manifest has no metadata.name")
		}
		if _, exists := out[name]; exists {
			return nil, fmt.Errorf("duplicate simulated Node manifest %q", name)
		}
		if node.Namespace != "" {
			return nil, fmt.Errorf("simulated Node %q must be cluster-scoped, not namespace %q", name, node.Namespace)
		}
		if stated := node.Labels[corev1.LabelHostname]; stated != "" && stated != name {
			return nil, fmt.Errorf("simulated Node %q has conflicting hostname label %q", name, stated)
		}
		loc := cluster.EndpointTopology{
			Hostname:  name,
			Rack:      node.Labels[rackLabel],
			Zone:      node.Labels[corev1.LabelTopologyZone],
			Region:    node.Labels[corev1.LabelTopologyRegion],
			NodeCount: 1,
		}
		if loc.Zone == "" || loc.Region == "" {
			return nil, fmt.Errorf("simulated Node %q must declare both %s and %s", name, corev1.LabelTopologyZone, corev1.LabelTopologyRegion)
		}
		out[name] = loc
	}
	if err := validateLocationHierarchy(valuesInNodeOrder(out)); err != nil {
		return nil, err
	}
	return out, nil
}

// overlayNodeLocation preserves the actual worker placement while supplying
// only explicitly declared location levels. Missing Node manifests never
// generate geographic labels.
func overlayNodeLocation(base cluster.EndpointTopology, locations map[string]cluster.EndpointTopology) (cluster.EndpointTopology, error) {
	if len(locations) == 0 || base.Hostname == "" {
		return base, nil
	}
	loc, exists := locations[base.Hostname]
	if !exists {
		return base, fmt.Errorf("worker on %s has no matching simulated Node manifest", base.Hostname)
	}
	if loc.Rack != "" && base.Rack == "" {
		return base, fmt.Errorf("worker on %s has Node manifest rack %q but its scenario declares no rack placement", base.Hostname, loc.Rack)
	}
	if base.Rack != "" && loc.Rack != "" && base.Rack != loc.Rack {
		return base, fmt.Errorf("worker on %s has kernel rack %q but Node manifest rack %q", base.Hostname, base.Rack, loc.Rack)
	}
	if base.Zone != "" && base.Zone != loc.Zone {
		return base, fmt.Errorf("worker on %s has zone %q but Node manifest zone %q", base.Hostname, base.Zone, loc.Zone)
	}
	if base.Region != "" && base.Region != loc.Region {
		return base, fmt.Errorf("worker on %s has region %q but Node manifest region %q", base.Hostname, base.Region, loc.Region)
	}
	base.Zone, base.Region = loc.Zone, loc.Region
	return base, nil
}

func valuesInNodeOrder(locations map[string]cluster.EndpointTopology) []cluster.EndpointTopology {
	names := make([]string, 0, len(locations))
	for name := range locations {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]cluster.EndpointTopology, 0, len(names))
	for _, name := range names {
		out = append(out, locations[name])
	}
	return out
}

// validateLocationHierarchy prevents one rack from being placed in two zones
// or one zone from being placed in two regions. Missing rack labels are allowed:
// the kernel may supply the rack when the Node manifest supplies only geography.
func validateLocationHierarchy(locations []cluster.EndpointTopology) error {
	hostLocation := map[string]cluster.EndpointTopology{}
	rackZone := map[string]string{}
	zoneRegion := map[string]string{}
	for _, loc := range locations {
		if loc.Hostname != "" {
			if prior, exists := hostLocation[loc.Hostname]; exists &&
				(prior.Rack != loc.Rack || prior.Zone != loc.Zone || prior.Region != loc.Region) {
				return fmt.Errorf("host %q has conflicting rack/zone/region placement", loc.Hostname)
			}
			hostLocation[loc.Hostname] = loc
		}
		if loc.Zone == "" && loc.Region == "" {
			continue
		}
		if loc.Zone == "" || loc.Region == "" {
			return fmt.Errorf("node %s has incomplete zone/region placement", loc.Hostname)
		}
		if prior, exists := zoneRegion[loc.Zone]; exists && prior != loc.Region {
			return fmt.Errorf("zone %q spans regions %q and %q", loc.Zone, prior, loc.Region)
		}
		zoneRegion[loc.Zone] = loc.Region
		if loc.Rack != "" {
			if prior, exists := rackZone[loc.Rack]; exists && prior != loc.Zone {
				return fmt.Errorf("rack %q spans zones %q and %q", loc.Rack, prior, loc.Zone)
			}
			rackZone[loc.Rack] = loc.Zone
		}
	}
	return nil
}

func podLocation(labels map[string]string) cluster.EndpointTopology {
	return cluster.EndpointTopology{
		Hostname: labels[corev1.LabelHostname],
		Rack:     labels[rackLabel],
		Zone:     labels[corev1.LabelTopologyZone],
		Region:   labels[corev1.LabelTopologyRegion],
	}
}

// manifestNodes returns the Node records of a deployment's manifests, which declare the zone and
// region of the simulated hosts (nodeLocations). Every other document is the router's to read
// (embed.Config.Manifests), which refuses a malformed one first; here it is skipped.
func manifestNodes(docs []byte) ([]*corev1.Node, error) {
	scheme := runtime.NewScheme()
	utilruntime.Must(corev1.AddToScheme(scheme))
	decoder := serializer.NewCodecFactory(scheme, serializer.EnableStrict).UniversalDeserializer()
	reader := k8syaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(docs)))
	var nodes []*corev1.Node
	for i := 0; ; i++ {
		doc, err := reader.Read()
		if err == io.EOF {
			return nodes, nil
		}
		if err != nil {
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		var tm metav1.TypeMeta
		if sigsyaml.Unmarshal(doc, &tm) != nil || tm.APIVersion != "v1" || tm.Kind != "Node" {
			continue
		}
		obj, _, err := decoder.Decode(doc, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		nodes = append(nodes, obj.(*corev1.Node))
	}
}
