// Original to blis-vllm.

package vllm

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
)

// Code BLIS does not have (design §11.4, code_not_visible). The vLLM release this build models (Version)
// loads a component from a
// module path, or by a name its in-tree registration does not hold, in these places
// (surface.json, category path_loader):
//
//   - kv_connector_module_path: the connector class is imported from it, ahead of the registry
//     (vllm/distributed/kv_transfer/kv_connector/factory.py:103-124@5df63a72);
//   - spec_module_path: the offloading spec, when spec_name is not registered
//     (vllm/v1/kv_offload/factory.py:30-51@5df63a72);
//   - secondary_tiers[].module_path: a secondary tier, when its type is not registered
//     (vllm/v1/kv_offload/tiering/factory.py:100-144@5df63a72);
//   - cache_policy_module_path: the CPU tier's eviction policy, when eviction_policy is not
//     registered (vllm/v1/kv_offload/cpu/policies/factory.py:38-81@5df63a72);
//   - a dotted backpressure_cls or policy_cls in a tier's backpressure block, and a dotted kvcr
//     tier policy (tiering/factory.py:20-35, tiering/kvcr/manager.py:100-118@5df63a72);
//   - EventPublisherFactory.register_publisher: an out-of-tree publisher, which a plugin
//     registers under a name KVEventsConfig's literal does not hold (vllm/distributed/kv_events.py:
//     566-598, vllm/config/kv_events.py:19@5df63a72);
//   - scheduler_cls, a qualified name the scheduler class is resolved from
//     (vllm/config/scheduler.py:228-253@5df63a72);
//   - speculative method custom_class, whose proposer is imported from the module path
//     speculative_config.model names (vllm/config/speculative.py:1110-1134,
//     vllm/v1/spec_decode/custom_class_proposer.py:30-40@5df63a72).
//
// A name the release does not register reaches the same code: vLLM refuses it unless something
// outside the release registered it first, so a run that states one ran code BLIS cannot see. Each
// is refused at start-up by the setting that names it, with a *CodeNotVisibleError, before any
// other refusal (§11.4: the reason takes precedence over field_not_modelled). A path vLLM does not
// read, such as spec_module_path beside a registered spec_name, loads nothing and is not refused
// for it.

// CodeNotVisibleError is a refusal of a setting that makes vLLM run code the pinned release does not
// ship. A host reads its reason with errors.As and G5Code.
type CodeNotVisibleError struct {
	// Setting is the setting as the engine block states it, such as
	// kv_transfer.kv_connector_module_path.
	Setting string
	// Value is the module path or name it states.
	Value string
	// Loads says what vLLM does with the value, with its citation.
	Loads string
}

func (e *CodeNotVisibleError) Error() string {
	return fmt.Sprintf("vllm: %s %q names code %s does not ship: %s; a run that loads it is not "+
		"simulated (code_not_visible, design §11.4)", e.Setting, e.Value, release, e.Loads)
}

// G5Code is the reason a point that states the setting is not scored.
func (e *CodeNotVisibleError) G5Code() pointrecord.G5Code { return pointrecord.CodeNotVisible }

// release names the vLLM release this build models in a refusal.
const release = "vLLM " + Version

// The names the release registers in-tree, as surface.json lists them
// (TestTheRegisteredNamesAreTheSurfaceList, which fails when the list is regenerated for a release
// that registers others).
var (
	// registeredConnectors: KVConnectorFactory.register_connector
	// (vllm/distributed/kv_transfer/kv_connector/factory.py:153-249@5df63a72).
	registeredConnectors = []string{"DecodeBenchConnector", "ExampleConnector", "ExampleHiddenStatesConnector",
		"FlexKVConnectorV1", "HF3FSKVConnector", "HiSparseConnector", "LMCacheConnectorV1", "LMCacheMPConnector",
		"MoRIIOConnector", "MooncakeConnector", "MooncakeStoreConnector", "MultiConnector", "NixlConnector",
		"NixlPullConnector", "NixlPushConnector", "OffloadingConnector", "SimpleCPUOffloadConnector"}
	// registeredSpecs: OffloadingSpecFactory.register_spec (vllm/v1/kv_offload/factory.py:62-68@5df63a72).
	registeredSpecs = []string{"CPUOffloadingSpec", "TieringOffloadingSpec"}
	// registeredTiers: SecondaryTierFactory.register_tier (vllm/v1/kv_offload/tiering/factory.py:146-174@4ba787ec).
	registeredTiers = []string{"example", "fs", "kvcr", "obj", "p2p"}
	// registeredPolicies: CachePolicyFactory.register_cache_policy
	// (vllm/v1/kv_offload/cpu/policies/factory.py:86-91@5df63a72).
	registeredPolicies = []string{"arc", "lru"}
	// registeredPublishers: EventPublisherFactory._registry (vllm/distributed/kv_events.py:567-570@5df63a72).
	registeredPublishers = []string{"null", "zmq"}
	// backpressureClasses: the classes a short backpressure_cls or policy_cls resolves to in
	// vllm/v1/kv_offload/tiering/backpressure.py@5df63a72 (tiering/factory.py:20-35).
	backpressureClasses = []string{"DropAccountingPolicy", "DropStorePolicy", "EMABackpressureDetector", "ThrottledDropPolicy"}
)

// refuseNotVisible refuses the first setting of the block that makes vLLM load code it does not ship.
func refuseNotVisible(c Config) error {
	if t := c.KVTransfer; t != nil {
		if err := connectorNotVisible("kv_transfer", t.Connector, t.ModulePath); err != nil {
			return err
		}
		if t.ExtraConfig != nil {
			for i, k := range t.ExtraConfig.Connectors {
				at := fmt.Sprintf("kv_transfer.kv_connector_extra_config.connectors[%d]", i)
				if err := connectorNotVisible(at, k.KVConnector, k.ModulePath); err != nil {
					return err
				}
				if x := k.ExtraConfig; x != nil {
					if err := offloadNotVisible(at+".kv_connector_extra_config", x.offloadKeys()); err != nil {
						return err
					}
				}
			}
		}
	}
	if o := c.KVOffload; o != nil {
		if err := connectorNotVisible("kv_offload", o.Connector, o.ModulePath); err != nil {
			return err
		}
		if err := offloadNotVisible("kv_offload", o.offloadKeys()); err != nil {
			return err
		}
	}
	if ev := c.KVEvents; ev != nil && ev.Publisher != "" && !slices.Contains(registeredPublishers, ev.Publisher) {
		return &CodeNotVisibleError{Setting: "kv_events_config.publisher", Value: ev.Publisher,
			Loads: release + " publishes through null or zmq only; another publisher is one a plugin registered with " +
				"EventPublisherFactory.register_publisher (vllm/distributed/kv_events.py:566-598@5df63a72)"}
	}
	if c.SchedulerCls != "" {
		// A name under vllm. is a class the release ships; this engine still does not read the setting.
		if !strings.HasPrefix(c.SchedulerCls, "vllm.") {
			return &CodeNotVisibleError{Setting: "scheduler_cls", Value: c.SchedulerCls,
				Loads: "vLLM imports the scheduler class from this qualified name (vllm/config/scheduler.py:228-253@5df63a72)"}
		}
		return fmt.Errorf("vllm: scheduler_cls %q is not modelled: the engine runs the Scheduler or AsyncScheduler that "+
			"async scheduling selects (%s)", c.SchedulerCls, issueBase+"66")
	}
	if sp := c.Speculative; sp != nil && (sp.Method == "custom_class" || (sp.Method == "" && customProposerPath(sp.DraftModel))) {
		return &CodeNotVisibleError{Setting: "speculative_config.model", Value: sp.DraftModel,
			Loads: "speculative method custom_class imports its proposer from this module path " +
				"(vllm/config/speculative.py:1110-1134, vllm/v1/spec_decode/custom_class_proposer.py:30-40@5df63a72)"}
	}
	return nil
}

// customProposerPath is SpeculativeConfig._is_custom_proposer_path (vllm/config/speculative.py:1110-1119@5df63a72):
// a dotted import path, not a URL or a file path.
func customProposerPath(model string) bool {
	if model == "" || strings.HasPrefix(model, "http://") || strings.HasPrefix(model, "https://") ||
		strings.HasPrefix(model, "file://") || strings.Contains(model, "/") {
		return false
	}
	parts := strings.Split(model, ".")
	if len(parts) < 2 {
		return false
	}
	for _, p := range parts {
		if !identifier(p) {
			return false
		}
	}
	return true
}

// identifier is Python's str.isidentifier for an ASCII name.
func identifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		letter := r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		if !letter && (i == 0 || r < '0' || r > '9') {
			return false
		}
	}
	return true
}

// connectorNotVisible refuses a connector loaded from a module path, or one the release does not
// register. vLLM imports from kv_connector_module_path ahead of its registry, so a stated path loads
// out-of-tree code whatever the name; a stated empty path vLLM refuses itself.
func connectorNotVisible(at, name string, modulePath *string) error {
	if modulePath != nil {
		if *modulePath == "" {
			return fmt.Errorf("vllm: %s.kv_connector_module_path is empty, which vLLM refuses "+
				"(vllm/distributed/kv_transfer/kv_connector/factory.py:104-105@5df63a72)", at)
		}
		return &CodeNotVisibleError{Setting: at + ".kv_connector_module_path", Value: *modulePath,
			Loads: fmt.Sprintf("vLLM imports connector %q from this module, ahead of its own registry "+
				"(vllm/distributed/kv_transfer/kv_connector/factory.py:103-124@5df63a72)", name)}
	}
	if name != "" && !slices.Contains(registeredConnectors, name) {
		key := ".connector"
		if strings.Contains(at, "connectors[") {
			key = ".kv_connector"
		}
		return &CodeNotVisibleError{Setting: at + key, Value: name,
			Loads: release + " registers no such connector, so only one an out-of-tree plugin registered could run " +
				"(vllm/distributed/kv_transfer/kv_connector/factory.py:125-131@5df63a72)"}
	}
	return nil
}

// offloadKeys are the OffloadingConnector's kv_connector_extra_config keys that choose code.
type offloadKeys struct {
	spec, specModulePath, policy, policyModulePath string
	tiers                                          []map[string]any
	backpressure                                   map[string]any
}

// offloadNotVisible refuses an offloading spec, eviction policy, secondary tier or backpressure class
// that vLLM would load from outside the release.
func offloadNotVisible(at string, k offloadKeys) error {
	spec := defaultOr(k.spec, "CPUOffloadingSpec")
	if !slices.Contains(registeredSpecs, spec) {
		loads := release + " registers no such spec, so only one a plugin registered could run"
		if k.specModulePath != "" {
			loads = fmt.Sprintf("vLLM imports spec %q from spec_module_path %q", spec, k.specModulePath)
		}
		return &CodeNotVisibleError{Setting: at + ".spec_name", Value: spec,
			Loads: loads + " (vllm/v1/kv_offload/factory.py:30-51@5df63a72)"}
	}
	// Both registered specs read eviction_policy and cache_policy_module_path (cpu/spec.py:134-137,
	// tiering/spec.py:365@5df63a72); the path is read only for a name the registry does not hold.
	if policy := defaultOr(k.policy, "lru"); !slices.Contains(registeredPolicies, policy) {
		loads := release + " registers no such eviction policy, so only one a plugin registered could run"
		if k.policyModulePath != "" {
			loads = fmt.Sprintf("vLLM imports policy %q from cache_policy_module_path %q", policy, k.policyModulePath)
		}
		return &CodeNotVisibleError{Setting: at + ".eviction_policy", Value: policy,
			Loads: loads + " (vllm/v1/kv_offload/cpu/policies/factory.py:38-81@5df63a72)"}
	}
	// Only TieringOffloadingSpec reads secondary_tiers and backpressure (tiering/spec.py:241-250,
	// 285-313@5df63a72); CPUOffloadingSpec ignores them.
	if spec != "TieringOffloadingSpec" {
		return nil
	}
	if err := backpressureNotVisible(at+".backpressure", k.backpressure); err != nil {
		return err
	}
	for i, t := range k.tiers {
		tat := fmt.Sprintf("%s.secondary_tiers[%d]", at, i)
		typ, _ := t["type"].(string)
		mp, _ := t["module_path"].(string)
		switch {
		case typ != "" && !slices.Contains(registeredTiers, typ) && mp != "":
			return &CodeNotVisibleError{Setting: tat + ".module_path", Value: mp,
				Loads: fmt.Sprintf("vLLM imports secondary tier %q from this module "+
					"(vllm/v1/kv_offload/tiering/factory.py:100-144@5df63a72)", typ)}
		case typ != "" && !slices.Contains(registeredTiers, typ):
			return &CodeNotVisibleError{Setting: tat + ".type", Value: typ,
				Loads: release + " registers no such secondary tier, so only one a plugin registered could run " +
					"(vllm/v1/kv_offload/tiering/factory.py:100-144@5df63a72)"}
		}
		if p, ok := t["policy"].(string); ok && typ == "kvcr" && strings.Contains(p, ".") {
			return &CodeNotVisibleError{Setting: tat + ".policy", Value: p,
				Loads: "the kvcr tier imports a dotted policy name (vllm/v1/kv_offload/tiering/kvcr/manager.py:100-118@5df63a72)"}
		}
		bp, _ := t["backpressure"].(map[string]any)
		if err := backpressureNotVisible(tat+".backpressure", bp); err != nil {
			return err
		}
	}
	return nil
}

// backpressureNotVisible refuses a backpressure_cls or policy_cls that is a dotted path, which vLLM
// imports, or a short name its backpressure module does not define (tiering/factory.py:20-35,
// 72-89@5df63a72).
func backpressureNotVisible(at string, bp map[string]any) error {
	keys := make([]string, 0, len(bp))
	for k := range bp {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if k != "backpressure_cls" && k != "policy_cls" {
			continue
		}
		name, _ := bp[k].(string)
		if name == "" || slices.Contains(backpressureClasses, name) {
			continue
		}
		loads := "vLLM imports a dotted class name from its module"
		if !strings.Contains(name, ".") {
			loads = "vllm/v1/kv_offload/tiering/backpressure.py defines no such class"
		}
		return &CodeNotVisibleError{Setting: at + "." + k, Value: name,
			Loads: loads + " (vllm/v1/kv_offload/tiering/factory.py:20-35@5df63a72)"}
	}
	return nil
}
