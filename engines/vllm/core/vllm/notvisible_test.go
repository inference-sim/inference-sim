// Original to blis-vllm.

package vllm

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
)

// offloadWith is an OffloadingConnector block with the given extra kv_offload keys.
func offloadWith(keys string) string {
	return "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, bytes_per_block: 4096" + keys + "}\n"
}

// childWith is a MultiConnector whose offloading child's kv_connector_extra_config carries keys.
func childWith(keys string) string {
	return "kv_transfer: {connector: MultiConnector, kv_role: kv_both, kv_connector_extra_config: {connectors: [" +
		"{kv_connector: NixlConnector, kv_role: kv_both}, {kv_connector: OffloadingConnector, kv_role: kv_both, " +
		"kv_connector_extra_config: {cpu_bytes_to_use: 819200" + keys + "}}]}}\n" + derivedTier
}

const tiering = ", spec: TieringOffloadingSpec"

// notVisibleCases are deployments that make the modelled vLLM release load code it does not ship, each with the
// setting the refusal names and the value. item is the surface.json path_loader item a case runs,
// where it is one.
var notVisibleCases = []struct {
	name, item, block, setting, value string
}{
	{"connector from a module path", "kv_transfer_config.kv_connector_module_path",
		"kv_transfer: {connector: NixlConnector, kv_role: kv_producer, kv_connector_module_path: my_pkg.conn}\n",
		"kv_transfer.kv_connector_module_path", "my_pkg.conn"},
	{"a registered connector name from a module path still loads the module", "",
		"kv_transfer: {connector: MooncakeConnector, kv_role: kv_producer, kv_connector_module_path: my_pkg.conn}\n",
		"kv_transfer.kv_connector_module_path", "my_pkg.conn"},
	{"offload connector from a module path", "",
		"kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, bytes_per_block: 4096, kv_connector_module_path: my_pkg.off}\n",
		"kv_offload.kv_connector_module_path", "my_pkg.off"},
	{"MultiConnector child from a module path", "",
		"kv_transfer: {connector: MultiConnector, kv_role: kv_both, kv_connector_extra_config: {connectors: [" +
			"{kv_connector: NixlConnector, kv_role: kv_both, kv_connector_module_path: my_pkg.nixl}, " +
			"{kv_connector: OffloadingConnector, kv_role: kv_both, kv_connector_extra_config: {cpu_bytes_to_use: 819200}}]}}\n" + derivedTier,
		"kv_transfer.kv_connector_extra_config.connectors[0].kv_connector_module_path", "my_pkg.nixl"},
	{"a connector no release registers", "",
		"kv_transfer: {connector: AcmeConnector, kv_role: kv_producer}\n",
		"kv_transfer.connector", "AcmeConnector"},
	{"spec from a module path", "kv_connector_extra_config.spec_module_path",
		offloadWith(", spec: MySpec, spec_module_path: my_pkg.spec"), "kv_offload.spec_name", "MySpec"},
	{"spec from a module path, in the MultiConnector child", "",
		childWith(", spec_name: MySpec, spec_module_path: my_pkg.spec"),
		"kv_transfer.kv_connector_extra_config.connectors[1].kv_connector_extra_config.spec_name", "MySpec"},
	{"a spec no release registers", "", offloadWith(", spec: MySpec"), "kv_offload.spec_name", "MySpec"},
	{"secondary tier from a module path", "kv_connector_extra_config.secondary_tiers[].module_path",
		offloadWith(tiering + ", secondary_tiers: [{type: MyTier, module_path: my_pkg.tier}]"),
		"kv_offload.secondary_tiers[0].module_path", "my_pkg.tier"},
	{"a secondary tier no release registers", "",
		offloadWith(tiering + ", secondary_tiers: [{type: MyTier}]"), "kv_offload.secondary_tiers[0].type", "MyTier"},
	{"eviction policy from a module path", "kv_connector_extra_config.cache_policy_module_path",
		offloadWith(", eviction_policy: MyPolicy, cache_policy_module_path: my_pkg.policy"),
		"kv_offload.eviction_policy", "MyPolicy"},
	{"eviction policy from a module path, in the MultiConnector child", "",
		childWith(", eviction_policy: MyPolicy, cache_policy_module_path: my_pkg.policy"),
		"kv_transfer.kv_connector_extra_config.connectors[1].kv_connector_extra_config.eviction_policy", "MyPolicy"},
	{"dotted backpressure class", "kv_connector_extra_config.secondary_tiers[].backpressure (dotted backpressure_cls or policy_cls)",
		offloadWith(tiering + ", secondary_tiers: [{type: fs, root_dir: /x, backpressure: {backpressure_cls: my_pkg.Detector}}]"),
		"kv_offload.secondary_tiers[0].backpressure.backpressure_cls", "my_pkg.Detector"},
	{"dotted backpressure policy at the top of the extra config", "",
		offloadWith(tiering + ", backpressure: {policy_cls: my_pkg.Policy}, secondary_tiers: [{type: fs, root_dir: /x}]"),
		"kv_offload.backpressure.policy_cls", "my_pkg.Policy"},
	{"dotted kvcr policy", "kv_connector_extra_config.secondary_tiers[] kvcr policy (dotted name)",
		offloadWith(tiering + ", secondary_tiers: [{type: kvcr, policy: my_pkg.KVPolicy}]"),
		"kv_offload.secondary_tiers[0].policy", "my_pkg.KVPolicy"},
	{"out-of-tree event publisher", "kv_events_config.publisher (EventPublisherFactory.register_publisher)",
		"kv_events_config: {enable_kv_cache_events: true, publisher: kafka}\n", "kv_events_config.publisher", "kafka"},
	{"scheduler class from a qualified name", "scheduler_config.scheduler_cls",
		"scheduler_cls: my_pkg.MyScheduler\n", "scheduler_cls", "my_pkg.MyScheduler"},
	{"custom_class proposer", "speculative_config.method custom_class (speculative_config.model as a module path)",
		"speculative_config: {method: custom_class, num_speculative_tokens: 2, model: my_pkg.Proposer}\n",
		"speculative_config.model", "my_pkg.Proposer"},
	{"custom_class inferred from a dotted model", "",
		"speculative_config: {num_speculative_tokens: 2, model: my_pkg.Proposer}\n",
		"speculative_config.model", "my_pkg.Proposer"},
}

func TestCodeVLLMLoadsFromOutsideTheReleaseIsRefusedByName(t *testing.T) {
	for _, c := range notVisibleCases {
		t.Run(c.name, func(t *testing.T) {
			_, err := build(t, c.block)()
			if err == nil {
				t.Fatal("built, want refused as code_not_visible")
			}
			var nv *CodeNotVisibleError
			if !errors.As(err, &nv) {
				t.Fatalf("refused for another reason: %v", err)
			}
			if nv.G5Code() != pointrecord.CodeNotVisible {
				t.Errorf("reason %q, want %q", nv.G5Code(), pointrecord.CodeNotVisible)
			}
			if nv.Setting != c.setting || nv.Value != c.value {
				t.Errorf("refusal names %s %q, want %s %q", nv.Setting, nv.Value, c.setting, c.value)
			}
			if !strings.Contains(err.Error(), c.value) || !strings.Contains(err.Error(), "code_not_visible") {
				t.Errorf("message does not name the value and the reason: %v", err)
			}
		})
	}
}

// What vLLM does not read loads nothing, so it is not refused for loading code: a module path beside
// a name the registry holds, and a secondary tier under CPUOffloadingSpec, which reads none.
func TestAPathVLLMDoesNotReadIsNotCodeNotVisible(t *testing.T) {
	for _, c := range []struct{ name, block string }{
		{"spec_module_path beside a registered spec", offloadWith(", spec_module_path: my_pkg.spec")},
		{"cache_policy_module_path beside a registered policy", offloadWith(", eviction_policy: lru, cache_policy_module_path: my_pkg.p")},
		{"a secondary tier with a module path under CPUOffloadingSpec", offloadWith(", secondary_tiers: [{type: MyTier, module_path: my_pkg.tier}]")},
		{"a registered publisher", "kv_events_config: {enable_kv_cache_events: true, publisher: zmq}\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := build(t, c.block)(); err != nil {
				t.Fatalf("refused: %v", err)
			}
		})
	}
}

// The kvcr and example tiers are registered in-tree (vllm/v1/kv_offload/tiering/factory.py:146-174
// @5df63a72); like fs, obj and p2p they are refused by type as not modelled, not as code BLIS lacks.
func TestEveryRegisteredSecondaryTierIsRefusedByItsType(t *testing.T) {
	for _, typ := range registeredTiers {
		t.Run(typ, func(t *testing.T) {
			_, err := build(t, offloadWith(tiering+", secondary_tiers: [{type: "+typ+"}]"))()
			if err == nil {
				t.Fatal("built, want refused")
			}
			var nv *CodeNotVisibleError
			if errors.As(err, &nv) {
				t.Fatalf("refused as code_not_visible, want not modelled: %v", err)
			}
			if !strings.Contains(err.Error(), `"`+typ+`"`) || !strings.Contains(err.Error(), "not modelled") ||
				!strings.Contains(err.Error(), "issues/63") {
				t.Errorf("refusal does not name the tier, say it is not modelled and cite its issue: %v", err)
			}
		})
	}
}

// surfaceItems are surface.json's items of a category.
func surfaceItems(t *testing.T, category string) []map[string]any {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "surface.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Items []map[string]any `json:"items"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, it := range doc.Items {
		if it["category"] == category {
			out = append(out, it)
		}
	}
	if len(out) == 0 {
		t.Fatalf("surface.json has no %s item", category)
	}
	return out
}

// Every path-loaded component the generated surface list holds is refused with code_not_visible, and
// a case above runs it, so a component a release adds cannot be listed without a refusal.
func TestEveryPathLoaderOfTheSurfaceListIsRefusedAndRun(t *testing.T) {
	run := map[string]bool{}
	for _, c := range notVisibleCases {
		if c.item != "" {
			run[c.item] = true
		}
	}
	for _, it := range surfaceItems(t, "path_loader") {
		name, _ := it["name"].(string)
		if it["status"] != "refused" || it["refusal"] != string(pointrecord.CodeNotVisible) {
			t.Errorf("%s: status %v refusal %v, want refused with code_not_visible", name, it["status"], it["refusal"])
		}
		if !run[name] {
			t.Errorf("%s: no case of notVisibleCases runs it", name)
		}
	}
}

// The names the engine takes as in-tree are the ones the generated surface list holds.
func TestTheRegisteredNamesAreTheSurfaceList(t *testing.T) {
	for _, c := range []struct {
		category string
		names    []string
	}{
		{"connector", registeredConnectors}, {"offload_spec", registeredSpecs}, {"secondary_tier", registeredTiers},
		{"cache_policy", registeredPolicies}, {"kv_event_publisher", registeredPublishers},
		{"backpressure_class", backpressureClasses},
	} {
		var want []string
		for _, it := range surfaceItems(t, c.category) {
			want = append(want, it["name"].(string))
		}
		sort.Strings(want)
		got := slices.Clone(c.names)
		sort.Strings(got)
		if !slices.Equal(got, want) {
			t.Errorf("%s: the engine takes %v as in-tree, surface.json lists %v", c.category, got, want)
		}
	}
}
