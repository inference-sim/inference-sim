// Original to blis-vllm.

package vllm

import (
	"bytes"
	"fmt"

	"github.com/inference-sim/blis-vllm/core/connector/multi"
	"github.com/inference-sim/blis-vllm/core/sched"
	"gopkg.in/yaml.v3"
)

// vLLM's MultiConnector through engine.Engine: a NIXL connector and the OffloadingConnector behind
// one connector, as llm-d's P/D with KV offload runs them
// (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:135@5df63a72). The engine block
// states it as vLLM does. kv_transfer names MultiConnector, and its kv_connector_extra_config
// carries "connectors": a list of child KVTransferConfig objects in the order vLLM tries them, each
// with its own kv_connector, kv_role and kv_connector_extra_config; vLLM builds each child with
// KVTransferConfig(**ktc) (multi_connector.py:235-240@5df63a72), and the wrapper's own kv_role and
// kv_load_failure_policy are the top-level object's (vllm/v1/core/sched/scheduler.py:176-194@5df63a72).
// So a MultiConnector config handed to vLLM is the one handed here:
//
//	kv_transfer:
//	  connector: MultiConnector
//	  kv_role: kv_both
//	  kv_connector_extra_config:
//	    connectors:
//	      - {kv_connector: NixlConnector, kv_role: kv_both}
//	      - {kv_connector: OffloadingConnector, kv_role: kv_both,
//	         kv_connector_extra_config: {cpu_bytes_to_use: 17179869184}}
//
// The one departure is the top-level key spelling, connector for kv_connector, which every
// kv_transfer block here uses. The offloading child's vLLM keys (cpu_bytes_to_use, block_size,
// eviction_policy, offload_prompt_only, spec_name) come from its own kv_connector_extra_config.
// What vLLM derives from the model rather than reads from the config (bytes_per_block, page_size,
// workers, copies) stays in the engine block's kv_offload, which a MultiConnector therefore still
// needs; stating a vLLM key in both places is refused. vLLM's kv_offloading_size cannot stand for the
// offloading child: it replaces kv_connector with the OffloadingConnector
// (vllm/config/vllm.py:1105-1130@5df63a72), so a NIXL connector stated beside kv_offload without
// MultiConnector is refused.

const multiConnector = "MultiConnector"

// offloadChild is the offloading child's kv_connector.
const offloadChild = "OffloadingConnector"

// multiExtraConfig is a MultiConnector's kv_connector_extra_config.
type multiExtraConfig struct {
	// Connectors are the children, first tried first (multi_connector.py:230-249@5df63a72).
	Connectors []childConfig `yaml:"connectors"`
}

// UnmarshalYAML decodes strictly, as the engine block does, so a key the engine does not model
// (NIXL's backends, a child's engine_id) is refused by name, under the key that held it.
func (m *multiExtraConfig) UnmarshalYAML(n *yaml.Node) error {
	raw, err := yaml.Marshal(n)
	if err != nil {
		return err
	}
	type plain multiExtraConfig
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode((*plain)(m)); err != nil {
		return fmt.Errorf("kv_connector_extra_config: %w", err)
	}
	return nil
}

// childConfig is one child's KVTransferConfig as the engine models it: vLLM's own field names.
// Any other KVTransferConfig field (engine_id, kv_port, kv_buffer_device, kv_connector_module_path,
// enable_permute_local_kv, kv_load_failure_policy) is not modelled and the strict decode of the
// engine block refuses it by name.
type childConfig struct {
	KVConnector string `yaml:"kv_connector"`
	KVRole      kvRole `yaml:"kv_role"`
	// ExtraConfig is the child's kv_connector_extra_config: the offloading child's CPU tier keys.
	// A NIXL child's extra config is not modelled, so any key there is refused.
	ExtraConfig *childExtra `yaml:"kv_connector_extra_config,omitempty"`
}

// childExtra holds the OffloadingConnector's kv_connector_extra_config keys the engine models, under
// vLLM's names (offloading/config.py:98-134, cpu/spec.py:96-139, kv_offload/base.py:617-623@5df63a72).
type childExtra struct {
	SpecName          string `yaml:"spec_name,omitempty"`
	CPUBytesToUse     int64  `yaml:"cpu_bytes_to_use,omitempty"`
	BlockSize         int    `yaml:"block_size,omitempty"`
	EvictionPolicy    string `yaml:"eviction_policy,omitempty"`
	OffloadPromptOnly *bool  `yaml:"offload_prompt_only,omitempty"`
}

// installer hands the scheduler the connector an engine's block built (Scheduler.SetConnector), or
// collects it for the MultiConnector that wraps it.
type installer func(c sched.Connector, conf sched.ConnectorConfig) error

// pair is kv_connector_extra_config.connectors resolved: the NIXL child, the offloading child, and
// whether the offloading child is listed first.
type pair struct {
	nixl, offload childConfig
	offloadFirst  bool
}

// childOrder resolves the children. The pair is the only one modelled, in either order; any other
// child is refused by its kv_connector.
func childOrder(c *connectorConfig) (pair, error) {
	var p pair
	var kids []childConfig
	if c.ExtraConfig != nil {
		kids = c.ExtraConfig.Connectors
	}
	if len(kids) != 2 {
		return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors has %d child(ren): a MultiConnector "+
			"is modelled over one NIXL connector and the OffloadingConnector, in the order vLLM should try them "+
			"(docs/multiconnector.md)", len(kids))
	}
	haveNixl, haveOff := false, false
	for i, k := range kids {
		switch {
		case k.KVConnector == offloadChild && !haveOff:
			p.offload, p.offloadFirst, haveOff = k, i == 0, true
		case nixlNames[k.KVConnector] && !haveNixl:
			p.nixl, haveNixl = k, true
		case k.KVConnector == offloadChild || nixlNames[k.KVConnector]:
			return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors: child %q listed twice; want one NIXL "+
				"connector and one %s", k.KVConnector, offloadChild)
		default:
			return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors: child kv_connector %q is not modelled "+
				"beside the offloading connector (modelled: NixlConnector, NixlPullConnector, %s; registry/connectors.yaml)",
				k.KVConnector, offloadChild)
		}
		// KVTransferConfig refuses a kv_connector without a kv_role (kv_transfer.py:120-124@5df63a72).
		switch k.KVRole {
		case roleProducer, roleConsumer, roleBoth:
		default:
			return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors: child %q kv_role %q: want kv_producer, "+
				"kv_consumer or kv_both (vLLM requires a kv_role beside a kv_connector, vllm/config/kv_transfer.py:120-124@5df63a72)",
				k.KVConnector, k.KVRole)
		}
	}
	if p.nixl.ExtraConfig != nil {
		return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors: the NIXL child's kv_connector_extra_config " +
			"is not modelled (its keys, such as backends, are not read by the scheduler side this engine models)")
	}
	if p.offload.KVRole != roleBoth {
		return p, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config.connectors: the %s child's kv_role %q: only kv_both is "+
			"modelled (the oracle runs it as kv_both)", offloadChild, p.offload.KVRole)
	}
	return p, nil
}

// checkConnectorPair refuses the combinations of kv_transfer and kv_offload that vLLM refuses or that
// are not modelled, and returns the kv_offload the engine runs. A NIXL connector beside an offload
// tier runs only as a MultiConnector, and a MultiConnector is modelled only over the pair it lists;
// its offloading child's vLLM keys, read from the child, complete the kv_offload block.
func checkConnectorPair(c *connectorConfig, off *offloadConfig) (*offloadConfig, error) {
	switch {
	case c == nil:
		return off, nil
	case c.Connector != multiConnector:
		if c.ExtraConfig != nil {
			return nil, fmt.Errorf("vllm: kv_transfer.kv_connector_extra_config is read only for connector %s (its connectors), "+
				"and this block states connector %q", multiConnector, c.Connector)
		}
		if off != nil {
			return nil, fmt.Errorf("vllm: kv_transfer connector %q beside kv_offload: vLLM runs a NIXL connector and an "+
				"offload tier together only as a %s (kv_offloading_size replaces kv_connector with the "+
				"OffloadingConnector, vllm/config/vllm.py:1105-1130@5df63a72); state kv_transfer.connector: %s "+
				"with kv_connector_extra_config.connectors: [NixlConnector, %s]", c.Connector, multiConnector, multiConnector, offloadChild)
		}
		return nil, nil
	case off == nil:
		return nil, fmt.Errorf("vllm: kv_transfer connector %s without kv_offload: the offloading child's bytes_per_block (and "+
			"workers, copies), which vLLM derives from the model, are stated by kv_offload, and a MultiConnector over NIXL "+
			"alone is not modelled (docs/feature-matrix.md)", multiConnector)
	}
	p, err := childOrder(c)
	if err != nil {
		return nil, err
	}
	merged := *off
	if merged.Connector == "" {
		merged.Connector = offloadChild
	}
	x := p.offload.ExtraConfig
	if x == nil {
		x = &childExtra{}
	}
	for _, d := range []struct {
		key    string
		stated bool
	}{{"cpu_bytes", off.CPUBytes != 0}, {"block_size", off.BlockSize != 0}, {"eviction_policy", off.EvictionPolicy != ""},
		{"offload_prompt_only", off.OffloadPromptOnly != nil}, {"spec", off.Spec != ""}} {
		if d.stated {
			return nil, fmt.Errorf("vllm: kv_offload.%s beside a MultiConnector: the offloading child's own "+
				"kv_connector_extra_config states it as vLLM does; state it there once", d.key)
		}
	}
	if x.CPUBytesToUse <= 0 {
		return nil, fmt.Errorf("vllm: the %s child's kv_connector_extra_config.cpu_bytes_to_use must be stated: vLLM requires it "+
			"(vllm/v1/kv_offload/cpu/spec.py:97-101@5df63a72)", offloadChild)
	}
	merged.Spec, merged.CPUBytes, merged.BlockSize = x.SpecName, x.CPUBytesToUse, x.BlockSize
	merged.EvictionPolicy, merged.OffloadPromptOnly = x.EvictionPolicy, x.OffloadPromptOnly
	return &merged, nil
}

// buildConnectors installs the engine's connector on the scheduler: the NIXL connector alone, the
// offloading connector alone, or both behind a MultiConnector in the order the block states.
func (e *Engine) buildConnectors(s *sched.Scheduler, st settings) error {
	hybrid := len(st.sched.KVGroups) > 1
	if st.connector == nil || st.connector.Connector != multiConnector {
		var err error
		if e.pd, err = buildConnector(s, st.connector, st.pp, st.sched.Spec, hybrid, e.at, s.SetConnector); err != nil {
			return err
		}
		e.gpu.pd = e.pd
		if st.offload != nil {
			if e.off, err = buildOffload(s, st.offload, st.sched.Spec, st.specMethod, st.sched.EnablePrefixCaching, e.at, s.SetConnector); err != nil {
				return err
			}
			e.gpu.off = e.off
		}
		return nil
	}
	kids, err := childOrder(st.connector)
	if err != nil {
		return err
	}
	nixlName, offloadFirst := kids.nixl.KVConnector, kids.offloadFirst
	var nixlConn, offConn sched.Connector
	var conf sched.ConnectorConfig
	nc := *st.connector
	// The NIXL child's own kv_role gates its transfers; the scheduler's producer and consumer flags
	// are the wrapper's (below).
	nc.Connector, nc.Role, nc.ExtraConfig = nixlName, kids.nixl.KVRole, nil
	switch st.connector.Role {
	case roleProducer, roleConsumer, roleBoth:
	default:
		return fmt.Errorf("vllm: kv_transfer.kv_role %q: want kv_producer, kv_consumer or kv_both", st.connector.Role)
	}
	if e.pd, err = buildConnector(s, &nc, st.pp, st.sched.Spec, hybrid, e.at, func(c sched.Connector, cf sched.ConnectorConfig) error {
		// The MultiConnector's own roles and load-failure policy are kv_transfer's (scheduler.py:176-194).
		nixlConn, conf = c, cf
		conf.Consumer, conf.Producer = st.connector.Role.consumer(), st.connector.Role.producer()
		return nil
	}); err != nil {
		return err
	}
	if e.off, err = buildOffload(s, st.offload, st.sched.Spec, st.specMethod, st.sched.EnablePrefixCaching, e.at,
		func(c sched.Connector, _ sched.ConnectorConfig) error { offConn = c; return nil }); err != nil {
		return err
	}
	children := []multi.Child{{Name: nixlName, Conn: nixlConn, Producer: e.pd.role.producer()},
		{Name: offloadChild, Conn: offConn, Producer: true}}
	e.pd.child, e.off.child = 0, 1
	if offloadFirst {
		children[0], children[1] = children[1], children[0]
		e.pd.child, e.off.child = 1, 0
	}
	m, err := multi.New(children)
	if err != nil {
		return fmt.Errorf("vllm: %w", err)
	}
	if err := s.SetConnector(m, conf); err != nil {
		return fmt.Errorf("vllm: %w", err)
	}
	e.pd.multi, e.off.multi = m, m
	// One sequence of transfer IDs, so TransferDone knows whose transfer ended.
	e.off.next = e.pd.next
	e.gpu.pd, e.gpu.off = e.pd, e.off
	return nil
}
