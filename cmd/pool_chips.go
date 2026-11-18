package cmd

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"

	"github.com/inference-sim/inference-sim/sim"
)

// Node pools of different chips (design SIM-17, #1925).
//
// A scenario states the chip of each pool in cluster.pools; cluster.hardware is the default for
// a pool that states none, and the one the run's own kernel is opened for. A bundle's node pool
// refers to the scenario's pool of the same name. An instance is priced and sized by the pool it
// is placed in, so for every distinct pool chip other than the scenario's own the run opens one
// variant: the same scenario with cluster.hardware set to that chip, one kernel and one engine
// factory for it. A chip with no catalog entry, or no coefficients, is refused by name.

// poolChip is one variant: the scenario opened for a pool chip other than the scenario's own.
type poolChip struct {
	chip     string // the pool's chip, as the scenario states it
	pool     string // the first node pool that carries it, for messages
	scenario string // the variant's file name in dir
	dir      string // a directory holding the variant alone
	model    *engineModel
}

// asVariant runs f with the run's scenario globals pointing at the variant, as the engine
// factory reads them, and restores them after.
func (c *poolChip) asVariant(f func()) {
	scenario, dir, opened := kernelScenario, kernelScenarioDir, kernelOpened
	kernelScenario, kernelScenarioDir, kernelOpened = c.scenario, c.dir, c.model
	defer func() { kernelScenario, kernelScenarioDir, kernelOpened = scenario, dir, opened }()
	f()
}

// writeChipVariant writes the scenario file at src with cluster.hardware set to chip, as the
// only file of a new directory, and returns the directory and the file's name.
func writeChipVariant(src, chip string) (dir, name string, err error) {
	raw, err := os.ReadFile(src)
	if err != nil {
		return "", "", err
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	var out bytes.Buffer
	enc := yaml.NewEncoder(&out)
	enc.SetIndent(2)
	set := false
	for {
		var doc yaml.Node
		if err := dec.Decode(&doc); err != nil {
			if err.Error() == "EOF" {
				break
			}
			return "", "", err
		}
		if root := doc.Content[0]; root.Kind == yaml.MappingNode && mappingValue(root, "kind") == "Scenario" {
			cluster := mappingNode(root, "cluster")
			if cluster == nil {
				return "", "", fmt.Errorf("scenario file %s states no cluster", src)
			}
			hw := mappingNode(cluster, "hardware")
			if hw == nil {
				return "", "", fmt.Errorf("scenario file %s states no cluster.hardware", src)
			}
			hw.Value = chip
			set = true
		}
		if err := enc.Encode(&doc); err != nil {
			return "", "", err
		}
	}
	if !set {
		return "", "", fmt.Errorf("scenario file %s holds no Scenario document", src)
	}
	if err := enc.Close(); err != nil {
		return "", "", err
	}
	dir, err = os.MkdirTemp("", "blis-pool-chip-")
	if err != nil {
		return "", "", err
	}
	name = strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)) + "@" + chip + ".yaml"
	return dir, name, os.WriteFile(filepath.Join(dir, name), out.Bytes(), 0o644)
}

// mappingNode is the value node of key in a YAML mapping, nil when absent.
func mappingNode(m *yaml.Node, key string) *yaml.Node {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

func mappingValue(m *yaml.Node, key string) string {
	if n := mappingNode(m, key); n != nil {
		return n.Value
	}
	return ""
}

// poolChipsWithoutBundle is the refusal for a scenario whose cluster.pools state a chip other than
// the cluster's (clusterChip) when the run has no node-pool bundle to place instances in them
// (inference-sim#2221). Without a bundle every instance is sized and priced as clusterChip, so the
// scenario's pool chips would be ignored silently. It is empty when every pool carries the
// cluster's chip, which the fallback prices correctly.
func poolChipsWithoutBundle(c scenario.Cluster, clusterChip string) string {
	var other []string
	for _, p := range c.Pools {
		if chip, _ := c.PoolHardware(p.Name); !strings.EqualFold(chip, clusterChip) {
			other = append(other, fmt.Sprintf("%s=%s", p.Name, chip))
		}
	}
	if len(other) == 0 {
		return ""
	}
	return fmt.Sprintf("states pool chips (cluster.pools: %s) that differ from cluster.hardware %q, but the run has no "+
		"node-pool bundle: without one every instance is sized and priced as %s. Give --policy-config a bundle with "+
		"node_pools naming each pool (examples/capabilities/node-pools-bundle.yaml), or state one chip for the whole cluster",
		strings.Join(other, ", "), clusterChip, clusterChip)
}

// openPoolChips checks each bundle node pool against the scenario's cluster and opens a variant
// for every distinct pool chip other than the scenario's own (nil when every pool carries it). A
// pool the scenario does not list, a pool whose gpu_type is not the scenario's chip for it, a
// chip the catalog or the coefficients cannot price, and a pool stating memory or a node size
// other than its chip's are fatal, each naming the pool and the chip.
func openPoolChips(pools []sim.NodePoolBundleConfig) map[string]*poolChip {
	if kernelScenario == "" {
		return nil
	}
	path := filepath.Join(kernelScenarioDir, kernelScenario)
	sc, _, err := contracts.LoadBundle(path)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if len(pools) == 0 {
		if msg := poolChipsWithoutBundle(sc.Cluster, gpu); msg != "" {
			logrus.Fatalf("scenario %q %s", kernelScenario, msg)
		}
		return nil
	}
	variants := map[string]*poolChip{}
	for _, np := range pools {
		chip, ok := sc.Cluster.PoolHardware(np.Name)
		if !ok {
			names := make([]string, len(sc.Cluster.Pools))
			for i, p := range sc.Cluster.Pools {
				names[i] = p.Name
			}
			logrus.Fatalf("policy bundle node pool %q is not a pool of scenario %q's cluster, which lists %s "+
				"(cluster.pools); the scenario states each pool's chip", np.Name, kernelScenario, strings.Join(names, ", "))
		}
		if !strings.EqualFold(np.GPUType, chip) {
			logrus.Fatalf("policy bundle node pool %q is %q GPUs, but scenario %q states chip %q for that pool "+
				"(cluster.pools; cluster.hardware where a pool states none): every instance is priced for its "+
				"pool's chip, so the bundle must agree with the scenario", np.Name, np.GPUType, kernelScenario, chip)
		}
		m := kernelOpened
		if !strings.EqualFold(chip, gpu) {
			v := variants[strings.ToLower(chip)]
			if v == nil {
				v = openPoolChip(path, np.Name, chip)
				variants[strings.ToLower(chip)] = v
			}
			m = v.model
		}
		// The kernel sizes every instance's KV from its chip's memory; a pool stating other
		// memory would describe a device the run does not simulate.
		if mem := m.Deployment().DeviceMemoryGiB; np.GPUMemoryGiB > 0 && math.Abs(np.GPUMemoryGiB-mem) > 1e-9 {
			logrus.Fatalf("policy bundle node pool %q states gpu_memory_gib %g, but the catalog chip %q has %g GiB, "+
				"from which the kernel sizes every instance's KV; state %g", np.Name, np.GPUMemoryGiB, chip, mem, mem)
		}
		// A pool's gpus_per_node is the node its chip's kernel places and prices ranks on.
		if gpn := m.Packaging().GPUsPerNode; np.GPUsPerNode != gpn {
			logrus.Fatalf("policy bundle node pool %q states gpus_per_node %d, but scenario %q resolves to %d GPUs "+
				"per node (%s) on chip %q, on which the kernel places and prices every rank; state %d",
				np.Name, np.GPUsPerNode, kernelScenario, gpn, m.Packaging(), chip, gpn)
		}
	}
	if len(variants) == 0 {
		return nil
	}
	return variants
}

// openPoolChip opens the scenario at path for chip, the chip of node pool pool.
func openPoolChip(path, pool, chip string) *poolChip {
	dir, name, err := writeChipVariant(path, chip)
	if err != nil {
		logrus.Fatalf("node pool %q, chip %q: %v", pool, chip, err)
	}
	if disaggregated := kernelPDOpened != nil || prefillInstances > 0 || decodeInstances > 0; disaggregated {
		logrus.Fatalf("node pool %q is chip %q, but scenario %q runs on %q: pools of different chips are "+
			"priced for colocated serving only; a prefill/decode run needs every pool on the scenario's chip "+
			"(inference-sim#2173)", pool, chip, kernelScenario, gpu)
	}
	m, err := openChipModel(name, dir)
	if err != nil {
		logrus.Fatalf("node pool %q: its chip %q cannot be priced: %v", pool, chip, err)
	}
	return &poolChip{chip: chip, pool: pool, scenario: name, dir: dir, model: m}
}

// withPoolChips returns the cluster's engine constructor, building each instance's engine from
// the variant of the chip its pool placed it on (cfg.GPU), or by def for the scenario's own
// chip. Each variant's factory is built over its own scenario, kernel and host overheads.
func withPoolChips(def func(int, sim.InstanceConfig) (*sim.EngineDriver, error), variants map[string]*poolChip,
	ref engineRef, horizon, seed int64, acceptance float64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	if len(variants) == 0 {
		return def
	}
	factories := map[string]func(int, sim.InstanceConfig) (*sim.EngineDriver, error){}
	keys := make([]string, 0, len(variants))
	for k := range variants {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		v := variants[k]
		v.asVariant(func() { factories[k] = engineDriverFactory(ref, horizon, seed, v.model, acceptance) })
	}
	return func(index int, cfg sim.InstanceConfig) (*sim.EngineDriver, error) {
		if f, ok := factories[strings.ToLower(cfg.GPU)]; ok {
			var (
				d   *sim.EngineDriver
				err error
			)
			variants[strings.ToLower(cfg.GPU)].asVariant(func() { d, err = f(index, cfg) })
			return d, err
		}
		return def(index, cfg)
	}
}
