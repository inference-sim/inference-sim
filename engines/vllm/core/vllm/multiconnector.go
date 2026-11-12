// Original to blis-vllm.

package vllm

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-vllm/core/connector/multi"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// vLLM's MultiConnector through engine.Engine: a NIXL connector (kv_transfer) and the
// OffloadingConnector (kv_offload) behind one connector, as llm-d's P/D with KV offload runs them
// (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:135@5df63a72). The engine block
// states it as vLLM does: kv_transfer names MultiConnector and the children in the order vLLM tries
// them, and kv_offload states the offload child's CPU tier. vLLM's kv_offloading_size cannot stand
// for it: it replaces kv_connector with the OffloadingConnector
// (vllm/config/vllm.py:1105-1130@5df63a72), so a NIXL connector stated beside kv_offload without
// MultiConnector is refused.

const multiConnector = "MultiConnector"

// offloadChild is the offloading child's name in kv_transfer.connectors.
const offloadChild = "OffloadingConnector"

// installer hands the scheduler the connector an engine's block built (Scheduler.SetConnector), or
// collects it for the MultiConnector that wraps it.
type installer func(c sched.Connector, conf sched.ConnectorConfig) error

// childOrder is kv_transfer.connectors resolved: the NIXL child's name, and whether the offload
// child is listed before it.
func childOrder(c *connectorConfig) (nixlName string, offloadFirst bool, err error) {
	if len(c.Connectors) != 2 {
		return "", false, fmt.Errorf("vllm: kv_transfer.connectors %v: a MultiConnector is modelled over one NIXL "+
			"connector and the OffloadingConnector, in the order vLLM should try them (docs/feature-matrix.md)", c.Connectors)
	}
	for i, name := range c.Connectors {
		switch {
		case name == offloadChild:
			offloadFirst = i == 0
		case nixlNames[name] && nixlName == "":
			nixlName = name
		default:
			return "", false, fmt.Errorf("vllm: kv_transfer.connectors: child %q is not modelled beside the offloading "+
				"connector (modelled: NixlConnector, NixlPullConnector, %s; registry/connectors.yaml)", name, offloadChild)
		}
	}
	if nixlName == "" || !slices.Contains(c.Connectors, offloadChild) {
		return "", false, fmt.Errorf("vllm: kv_transfer.connectors %v: want one NIXL connector and %s", c.Connectors, offloadChild)
	}
	return nixlName, offloadFirst, nil
}

// checkConnectorPair refuses the combinations of kv_transfer and kv_offload that vLLM refuses or that
// are not modelled. A NIXL connector beside an offload tier runs only as a MultiConnector, and a
// MultiConnector is modelled only over the pair it lists.
func checkConnectorPair(c *connectorConfig, off *offloadConfig) error {
	switch {
	case c == nil:
		return nil
	case c.Connector != multiConnector:
		if len(c.Connectors) > 0 {
			return fmt.Errorf("vllm: kv_transfer.connectors is the child list of connector %s, and this block states "+
				"connector %q", multiConnector, c.Connector)
		}
		if off != nil {
			return fmt.Errorf("vllm: kv_transfer connector %q beside kv_offload: vLLM runs a NIXL connector and an "+
				"offload tier together only as a %s (kv_offloading_size replaces kv_connector with the "+
				"OffloadingConnector, vllm/config/vllm.py:1105-1130@5df63a72); state kv_transfer.connector: %s "+
				"with connectors: [NixlConnector, %s]", c.Connector, multiConnector, multiConnector, offloadChild)
		}
		return nil
	case off == nil:
		return fmt.Errorf("vllm: kv_transfer connector %s without kv_offload: the offloading child's CPU tier is "+
			"stated by kv_offload, and a MultiConnector over NIXL alone is not modelled (docs/feature-matrix.md)", multiConnector)
	}
	_, _, err := childOrder(c)
	return err
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
	nixlName, offloadFirst, err := childOrder(st.connector)
	if err != nil {
		return err
	}
	var nixlConn, offConn sched.Connector
	var conf sched.ConnectorConfig
	nc := *st.connector
	nc.Connector, nc.Connectors = nixlName, nil
	if e.pd, err = buildConnector(s, &nc, st.pp, st.sched.Spec, hybrid, e.at, func(c sched.Connector, cf sched.ConnectorConfig) error {
		// The MultiConnector's own roles and load-failure policy are kv_transfer's (scheduler.py:176-194).
		nixlConn, conf = c, cf
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
