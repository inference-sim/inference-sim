package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Table DC4.1 (M3.2 row 9): the two vLLM recipes for Mistral-Large-4 and Inkling-2-Small, simulated
// on every NVIDIA hardware and strategy they serve, with every preset the recipe lists.
//
// The recipes are read from the vLLM tree at VLLM_SOURCE (recipes/models), the commit the engine
// models, so a recipe change shows up at the next engine upgrade as a changed, unhandled or refused
// scenario here. Each case is a scenario written from the recipe's resolved command: the base
// arguments and environment, the strategy's layout flags, the strategy and hardware overrides, the
// feature arguments, and the preset's own arguments and role edits, all in vLLM's names (X1). The
// test turns that command into a scenario and a deployment, runs it, and checks three things:
//
//   - the run completes every request and drops none as unservable, and every coefficient the run
//     selected states its method (measured, assumed, vendor_spec, ...), so the figures that rest on
//     a stand-in say so;
//   - every argument and environment variable of the command has a status in
//     engines/vllm/fields.json (VLM-3), launcher-only ones included (--no-numa-bind, --port,
//     --headless, GLOO_SOCKET_IFNAME); an argument the test does not know fails it, naming the
//     argument, so a recipe that gains one is not run with the argument silently dropped;
//   - the opt-in features run both off and on: the parsers (reasoning, tool calling) are
//     front-end only and must price nothing, DSpark (greedy and probabilistic drafting) and the
//     text-only switch run as their own arms.
//
// A preset is refused only where vLLM at the modelled commit refuses it. One layout is refused by
// the engine's own coverage and not by vLLM: pipeline parallelism over a model whose graph states
// its caches (the kernel sizes the stated caches over the whole stack, not per stage), which is
// Mistral on two 8-GPU H200 nodes as TP8 x PP2. The test names that message, so the day the kernel
// prices it the expectation fails and must be removed.

// recipesCommit is the vLLM commit the recipes are read at, the engine's pin
// (engines/vllm/fields.json "engine").
const recipesCommit = "0fda86a35419"

// expectedRefusals are the commands the engine refuses at the modelled commit, with the part of
// the message that names why. The command is run in its plain arm and must fail with it, so the day
// the engine accepts one the expectation fails and the entry goes.
//
//   - vLLM's own refusals: the model does not fit or does not divide. Mistral FP8 (about 1 TB of
//     weights) on one 4-GPU GB300 node leaves no room for the 262144-token window's KV, and as a
//     prefill or decode pool it leaves none at all; Inkling's 8 KV heads do not divide TP16, which
//     is two 8-GPU nodes of multi-node TP (the recipe's two-tray TP8 is GB200's).
var expectedRefusals = map[string]string{
	"mistral/gb300/single_node_tp/fp8":   "To serve at least one request with the model's max seq len",
	"mistral/gb300/pd_cluster/fp8":       "No available memory for the cache blocks",
	"inkling/b200/multi_node_tp/default": "does not divide the 8 KV heads",
	"inkling/b300/multi_node_tp/default": "does not divide the 8 KV heads",
}

// recipeAcceptance is the acceptance the recipes state for their drafters, which the run pins
// because the registry holds none for these models: Mistral's DSpark drafter is depth 3 with AL
// 2.81 greedy and 2.80 probabilistic (Mistral recipe, the TP8 DCP8 runbook and the probabilistic
// mode note), Inkling's depth 7 with AL 4.5 (Inkling recipe, guide).
var recipeAcceptance = map[string]struct {
	greedy, probabilistic float64
	source                string
}{
	"mistral": {2.81, 2.80, "vLLM recipe Mistral-Large-4-1T-A52B at " + recipesCommit + " (preset runbook gb300_dcp8, AL 2.81; mode note dspark_probabilistic, AL 2.80), measured there on GB300 with DSpark depth 3"},
	"inkling": {4.5, 4.5, "vLLM recipe Inkling-2-Small at " + recipesCommit + " (guide, DSpark acceptance length 4.5), measured there on H200 TP4"},
}

// recipeHardware is each NVIDIA hardware the recipes name, as the catalog and the cluster state it.
var recipeHardware = map[string]struct {
	catalog, fabric string
	perNode, rack   int
	family          string // the recipes' hardware_overrides key
	sm              string // arch:sm10x group in supported_hardware
}{
	"gb200": {"gb200-nvl72", "ib-400g", 4, 72, "blackwell", "arch:sm100"},
	"gb300": {"gb300", "ib-800g", 4, 72, "blackwell", "arch:sm103"},
	"b200":  {"b200", "ib-400g", 8, 0, "blackwell", "arch:sm100"},
	"b300":  {"b300", "ib-800g", 8, 0, "blackwell", "arch:sm103"},
	"h200":  {"h200", "ib-400g", 8, 0, "hopper", ""},
}

// recipeModel is the part of a vLLM recipe this test reads.
type recipeModel struct {
	Meta struct {
		Hardware map[string]string `yaml:"hardware"`
	} `yaml:"meta"`
	Model struct {
		ModelID  string            `yaml:"model_id"`
		BaseArgs []string          `yaml:"base_args"`
		BaseEnv  map[string]string `yaml:"base_env"`
	} `yaml:"model"`
	Features map[string]struct {
		Args        []string `yaml:"args"`
		Strategies  []string `yaml:"strategies"`
		DefaultMode string   `yaml:"default_mode"`
		Modes       map[string]struct {
			Args []string `yaml:"args"`
		} `yaml:"modes"`
	} `yaml:"features"`
	OptIn    []string `yaml:"opt_in_features"`
	Variants map[string]struct {
		ModelID           string   `yaml:"model_id"`
		SupportedHardware []string `yaml:"supported_hardware"`
	} `yaml:"variants"`
	Compatible        []string                     `yaml:"compatible_strategies"`
	StrategyHardware  map[string]map[string]string `yaml:"strategy_hardware"`
	StrategyMinGPUs   map[string]int               `yaml:"strategy_min_gpus"`
	StrategyOverrides map[string]struct {
		TP                int               `yaml:"tp"`
		Env               map[string]string `yaml:"env"`
		HardwareOverrides map[string]struct {
			ExtraArgs []string `yaml:"extra_args"`
		} `yaml:"hardware_overrides"`
		Prefill recipeRole `yaml:"prefill"`
		Decode  recipeRole `yaml:"decode"`
	} `yaml:"strategy_overrides"`
	Presets map[string]recipePreset `yaml:"presets"`
}

type recipeRole struct {
	Nodes       int                       `yaml:"nodes"`
	Parallelism string                    `yaml:"parallelism"`
	Args        []string                  `yaml:"args"`
	RemoveArgs  []string                  `yaml:"remove_args"`
	JSONArgs    map[string]map[string]any `yaml:"json_args"`
}

type recipePreset struct {
	Hardware string   `yaml:"hardware"`
	Strategy string   `yaml:"strategy"`
	Variants []string `yaml:"variants"`
	Select   struct {
		Nodes    int            `yaml:"nodes"`
		Features map[string]any `yaml:"features"`
	} `yaml:"select"`
	Args  []string              `yaml:"args"`
	Env   map[string]string     `yaml:"env"`
	Roles map[string]recipeRole `yaml:"roles"`
}

// recipeCase is one runnable command of a recipe: a hardware, a strategy or preset, a variant.
type recipeCase struct {
	family   string // "mistral" or "inkling"
	name     string
	hw       string
	variant  string
	model    string // catalog name
	strategy string
	pools    []casePool
	noSpec   bool // the preset or strategy turns speculative decoding off
	textOnly bool // the preset requires text_only
}

type casePool struct {
	role  string // colocated, prefill, decode
	nodes int
	args  []string
	env   map[string]string
	// workerArgs are the flags a non-head node of the pool adds (--headless).
	workerArgs []string
}

// recipeSource is the vLLM tree the recipes are read from, with the convention the engine's source
// checks use: skipped when VLLM_SOURCE is unset, a failure when VLLM_SOURCE_REQUIRED is set.
func recipeSource(t *testing.T) string {
	t.Helper()
	root := os.Getenv("VLLM_SOURCE")
	skip := t.Skipf
	if os.Getenv("VLLM_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("VLLM_SOURCE is unset: the Mistral-Large-4 and Inkling-2-Small recipe presets were NOT simulated")
	}
	if _, err := os.Stat(filepath.Join(root, "recipes", "models")); err != nil {
		skip("VLLM_SOURCE=%s has no recipes/models: the recipe presets were NOT simulated", root)
	}
	return root
}

func loadRecipe(t *testing.T, root, rel string) *recipeModel {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, "recipes", "models", rel))
	if err != nil {
		t.Fatalf("reading the recipe: %v", err)
	}
	var r recipeModel
	if err := yaml.Unmarshal(b, &r); err != nil {
		t.Fatalf("parsing %s: %v", rel, err)
	}
	return &r
}

// supports says whether a variant serves a hardware: named, or by its arch:sm10x group.
func (r *recipeModel) supports(variant, hw string) bool {
	for _, s := range r.Variants[variant].SupportedHardware {
		if s == hw || (s != "" && s == recipeHardware[hw].sm) || (s == "arch:sm100" && recipeHardware[hw].sm == "arch:sm100") {
			return true
		}
	}
	return false
}

func (r *recipeModel) catalogName(variant string) string {
	id := r.Variants[variant].ModelID
	if id == "" {
		id = r.Model.ModelID
	}
	return strings.ToLower(id[strings.LastIndex(id, "/")+1:])
}

// offers says whether a feature is offered on a strategy: it states none, or names it.
func (r *recipeModel) offers(feature, strategy string) bool {
	list := r.Features[feature].Strategies
	if len(list) == 0 {
		return true
	}
	for _, s := range list {
		if s == strategy {
			return true
		}
	}
	return false
}

// featureArgs are the arguments of a feature in one mode ("" is its default mode).
func (r *recipeModel) featureArgs(feature, mode string) []string {
	f := r.Features[feature]
	if len(f.Modes) == 0 {
		return f.Args
	}
	if mode == "" {
		mode = f.DefaultMode
	}
	return f.Modes[mode].Args
}

// pdConnector is the strategy pd_cluster's NIXL connector, which the recipes do not state: the
// renderer's pd_cluster strategy carries it (vllm-project/recipes strategies/pd_cluster.yaml,
// connectors.nixl at 9e64957b57de), and a preset edits it with json_args. The environment is the
// strategy's per role.
var pdConnector = map[string]struct {
	cfg map[string]any
	env map[string]string
	arg []string
}{
	"prefill": {
		cfg: map[string]any{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_load_failure_policy": "fail"},
		env: map[string]string{"VLLM_NIXL_SIDE_CHANNEL_PORT": "5557", "VLLM_NIXL_SIDE_CHANNEL_HOST": "$PREFILL_NODE_1",
			"NCCL_CUMEM_ENABLE": "1", "NCCL_MNNVL_ENABLE": "1", "NCCL_NVLS_ENABLE": "1", "UCX_NET_DEVICES": "all"},
		arg: []string{"--port", "8001"},
	},
	"decode": {
		cfg: map[string]any{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_load_failure_policy": "fail"},
		env: map[string]string{"VLLM_NIXL_SIDE_CHANNEL_PORT": "5558", "VLLM_NIXL_SIDE_CHANNEL_HOST": "$DECODE_NODE_1"},
		arg: []string{"--port", "8002", "--compilation-config", `{"cudagraph_mode":"FULL_DECODE_ONLY"}`},
	},
}

// poolArgs is the strategy's own layout flags for one pool: what the renderer's command synthesis
// emits for the strategy (vllm-project/recipes strategies/*.yaml and its command synthesis at
// 9e64957b57de): tensor parallelism over the GPUs, multi-node rendezvous, data parallelism with
// expert parallelism for DEP, tensor parallelism per node and pipeline parallelism across nodes.
func poolArgs(strategy string, nodes, gpn, tp int) (args, worker []string, env map[string]string) {
	rendezvous := func() {
		args = append(args, "--nnodes", strconv.Itoa(nodes), "--node-rank", "0", "--master-addr", "$HEAD_NODE")
		worker = []string{"--headless"}
	}
	env = map[string]string{"GLOO_SOCKET_IFNAME": "$IFACE_NAME", "NCCL_SOCKET_IFNAME": "$IFACE_NAME"}
	switch strategy {
	case "single_node_tp":
		if tp == 0 {
			tp = nodes * gpn
		}
		args = []string{"--tensor-parallel-size", strconv.Itoa(tp)}
	case "multi_node_tp":
		args = []string{"--tensor-parallel-size", strconv.Itoa(nodes * gpn)}
		rendezvous()
	case "multi_node_tp_pp":
		args = []string{"--tensor-parallel-size", strconv.Itoa(gpn), "--pipeline-parallel-size", strconv.Itoa(nodes)}
		rendezvous()
	case "multi_node_dep", "dep":
		args = []string{"--data-parallel-size", strconv.Itoa(nodes * gpn), "--data-parallel-size-local", strconv.Itoa(gpn),
			"--data-parallel-start-rank", "0", "--data-parallel-address", "$HEAD_NODE", "--data-parallel-rpc-port", "$DP_RPC_PORT",
			"--enable-expert-parallel"}
		if strategy == "multi_node_dep" {
			args = append(args, "--data-parallel-hybrid-lb")
		} else if nodes > 1 {
			args = append(args, "--data-parallel-hybrid-lb")
		}
	case "tp":
		args = []string{"--tensor-parallel-size", strconv.Itoa(nodes * gpn)}
		if nodes > 1 {
			rendezvous()
		}
	}
	return args, worker, env
}

// flagList is an argv read as ordered (flag, value) pairs; a flag with no value is a switch.
type flagPair struct{ name, val string }

func parseArgv(argv []string) []flagPair {
	var out []flagPair
	for i := 0; i < len(argv); i++ {
		if !strings.HasPrefix(argv[i], "--") {
			continue
		}
		p := flagPair{name: argv[i]}
		if i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "--") {
			p.val = argv[i+1]
			i++
		}
		out = append(out, p)
	}
	return out
}

// settle is the argv's value per flag, a later occurrence winning as vLLM's parser does.
func settle(argv []string, remove []string) (map[string]string, []string) {
	m := map[string]string{}
	var order []string
	for _, p := range parseArgv(argv) {
		if _, ok := m[p.name]; !ok {
			order = append(order, p.name)
		}
		m[p.name] = p.val
	}
	for _, r := range remove {
		delete(m, r)
	}
	kept := order[:0]
	for _, n := range order {
		if _, ok := m[n]; ok {
			kept = append(kept, n)
		}
	}
	return m, kept
}

// mergeJSONArgs applies a preset's json_args to a role's JSON-valued flag the way the renderer
// does: the map under the connector's name edits that connector's config, a null deleting the key.
func mergeJSONArgs(cfg map[string]any, edits map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range cfg {
		out[k] = v
	}
	for k, v := range edits {
		if v == nil {
			delete(out, k)
		} else {
			out[k] = v
		}
	}
	return out
}

// buildCases resolves every NVIDIA command a recipe offers: each strategy on each hardware it
// supports, in each variant that serves the hardware, then each NVIDIA preset.
func buildCases(t *testing.T, r *recipeModel, family string) []recipeCase {
	t.Helper()
	var cases []recipeCase
	variants := make([]string, 0, len(r.Variants))
	for v := range r.Variants {
		variants = append(variants, v)
	}
	sort.Strings(variants)
	feat := func(name string) []string { return r.featureArgs(name, "") }
	_ = feat
	// base is the arguments every command starts from: the model's base arguments and environment.
	base := func(extra ...string) ([]string, map[string]string) {
		env := map[string]string{}
		for k, v := range r.Model.BaseEnv {
			env[k] = v
		}
		return append(append([]string{}, r.Model.BaseArgs...), extra...), env
	}
	strategyOverride := func(strategy, hw string) (tp int, extra []string, env map[string]string) {
		so := r.StrategyOverrides[strategy]
		return so.TP, so.HardwareOverrides[recipeHardware[hw].family].ExtraArgs, so.Env
	}
	// the strategies a recipe lists, in a fixed order; a strategy_hardware entry marks the support.
	// strategy_hardware names either hardware (Mistral: the hardware a strategy is offered on) or
	// a family with "unsupported" (Inkling: hopper cannot run multi-node TP); a recipe that lists
	// compatible_strategies offers each on every hardware its variants serve.
	strategies := append([]string{}, r.Compatible...)
	if len(strategies) == 0 {
		for s := range r.StrategyHardware {
			strategies = append(strategies, s)
		}
	}
	sort.Strings(strategies)
	hardwareOf := func(strategy string) []string {
		var hws []string
		byHardware := false
		for key := range r.StrategyHardware[strategy] {
			if _, ok := recipeHardware[key]; ok {
				byHardware = true
			}
		}
		for hw, h := range recipeHardware {
			level := r.StrategyHardware[strategy][hw]
			if byHardware && level != "supported" {
				continue
			}
			if r.StrategyHardware[strategy][h.family] == "unsupported" {
				continue
			}
			hws = append(hws, hw)
		}
		sort.Strings(hws)
		return hws
	}
	nodesFor := func(strategy, hw string) int {
		switch strategy {
		case "single_node_tp":
			return 1
		case "pd_cluster":
			return 2
		}
		return 2
	}
	for _, strategy := range strategies {
		for _, hw := range hardwareOf(strategy) {
			h := recipeHardware[hw]
			for _, variant := range variants {
				if !r.supports(variant, hw) {
					continue
				}
				c := recipeCase{family: family, name: strategy, hw: hw, variant: variant, model: r.catalogName(variant), strategy: strategy}
				nodes := nodesFor(strategy, hw)
				if min := r.StrategyMinGPUs[hw]; min > nodes*h.perNode && strategy != "single_node_tp" {
					t.Fatalf("%s on %s needs %d GPUs and the layout has %d", strategy, hw, min, nodes*h.perNode)
				}
				if strategy == "pd_cluster" {
					so := r.StrategyOverrides[strategy]
					for _, role := range []string{"prefill", "decode"} {
						spec := so.Prefill
						if role == "decode" {
							spec = so.Decode
						}
						c.pools = append(c.pools, pdPool(r, role, spec, recipeRole{}, h.perNode, base))
					}
					cases = append(cases, c)
					continue
				}
				tp, extra, env := strategyOverride(strategy, hw)
				args, env0 := base(extra...)
				layout, worker, lenv := poolArgs(strategy, nodes, h.perNode, tp)
				for k, v := range lenv {
					env0[k] = v
				}
				for k, v := range env {
					env0[k] = v
				}
				c.pools = []casePool{{role: "colocated", nodes: nodes, args: append(append([]string{}, args...), layout...), env: env0, workerArgs: worker}}
				if strategy == "multi_node_tp_pp" {
					c.noSpec = true // spec_decoding is not offered for this strategy (features.spec_decoding.strategies)
				}
				cases = append(cases, c)
			}
		}
	}
	names := make([]string, 0, len(r.Presets))
	for n := range r.Presets {
		names = append(names, n)
	}
	sort.Strings(names)
	nvidia := 0
	for _, name := range names {
		p := r.Presets[name]
		h, ok := recipeHardware[p.Hardware]
		if !ok {
			if !strings.HasPrefix(p.Hardware, "mi") {
				t.Fatalf("preset %s is on hardware %q, which is neither NVIDIA hardware the test knows nor AMD", name, p.Hardware)
			}
			continue
		}
		nvidia++
		vs := p.Variants
		if len(vs) == 0 {
			vs = variants
		}
		for _, variant := range vs {
			if !r.supports(variant, p.Hardware) {
				continue
			}
			c := recipeCase{family: family, name: name, hw: p.Hardware, variant: variant, model: r.catalogName(variant), strategy: p.Strategy}
			if f, ok := p.Select.Features["spec_decoding"]; ok && f == false {
				c.noSpec = true
			}
			if f, ok := p.Select.Features["text_only"]; ok && f == true {
				c.textOnly = true
			}
			if p.Strategy == "pd_cluster" {
				for _, role := range []string{"prefill", "decode"} {
					c.pools = append(c.pools, pdPool(r, role, r.StrategyOverrides["pd_cluster"].Prefill, p.Roles[role], h.perNode, base))
					if role == "decode" {
						c.pools[len(c.pools)-1] = pdPool(r, role, r.StrategyOverrides["pd_cluster"].Decode, p.Roles[role], h.perNode, base)
					}
				}
				for i := range c.pools {
					for k, v := range p.Env {
						c.pools[i].env[k] = v
					}
				}
				cases = append(cases, c)
				continue
			}
			nodes := p.Select.Nodes
			if nodes == 0 {
				nodes = nodesFor(p.Strategy, p.Hardware)
			}
			tp, extra, env := strategyOverride(p.Strategy, p.Hardware)
			args, env0 := base(extra...)
			layout, worker, lenv := poolArgs(p.Strategy, nodes, h.perNode, tp)
			for k, v := range lenv {
				env0[k] = v
			}
			for k, v := range env {
				env0[k] = v
			}
			for k, v := range p.Env {
				env0[k] = v
			}
			c.pools = []casePool{{role: "colocated", nodes: nodes, args: append(append(append([]string{}, args...), layout...), p.Args...), env: env0, workerArgs: worker}}
			cases = append(cases, c)
		}
	}
	if family == "mistral" && nvidia != 8 {
		t.Errorf("the Mistral recipe lists %d NVIDIA presets, the test was written for 8: handle the new ones", nvidia)
	}
	return cases
}

// pdPool is one role of a pd_cluster command: the pool parallelism (dep: data parallel over the
// node's GPUs with expert parallelism; tp: tensor parallel over them), the role's port and
// compilation config, the NIXL connector with the preset's edits, and the preset's own role
// arguments and removals.
func pdPool(r *recipeModel, role string, so recipeRole, pr recipeRole, gpn int, base func(...string) ([]string, map[string]string)) casePool {
	nodes := so.Nodes
	if pr.Nodes != 0 {
		nodes = pr.Nodes
	}
	par := so.Parallelism
	if pr.Parallelism != "" {
		par = pr.Parallelism
	}
	args, env := base()
	layout, worker, lenv := poolArgs(par, nodes, gpn, 0)
	conn := pdConnector[role]
	cfg := conn.cfg
	for _, edit := range pr.JSONArgs["--kv-transfer-config"] {
		if m, ok := edit.(map[string]any); ok {
			cfg = mergeJSONArgs(cfg, m)
		}
	}
	kv, _ := json.Marshal(cfg)
	args = append(args, layout...)
	args = append(args, conn.arg...)
	args = append(args, "--kv-transfer-config", string(kv))
	args = append(args, pr.Args...)
	for k, v := range lenv {
		env[k] = v
	}
	for k, v := range conn.env {
		env[k] = v
	}
	// remove_args is final: drop the flag and its value.
	m, order := settle(args, pr.RemoveArgs)
	var out []string
	for _, n := range order {
		out = append(out, n)
		if m[n] != "" {
			out = append(out, m[n])
		}
	}
	return casePool{role: role, nodes: nodes, args: out, env: env, workerArgs: worker}
}

// arm is one feature setting of a command: the extra arguments the recipe's features add.
type arm struct {
	name     string
	feature  func(r *recipeModel) []string
	spec     string // "" or the recipe's mode name
	checksum bool   // metrics must equal the plain arm's (front-end only)
}

var recipeArms = []arm{
	{name: "plain"},
	{name: "parsers", feature: func(r *recipeModel) []string {
		return append(append([]string{}, r.featureArgs("reasoning", "")...), r.featureArgs("tool_calling", "")...)
	}, checksum: true},
	{name: "dspark", spec: "dspark"},
	{name: "dspark-probabilistic", spec: "dspark_probabilistic"},
	{name: "text-only", feature: func(r *recipeModel) []string { return r.featureArgs("text_only", "") }},
}

// translated is a command read as the scenario and deployment settings it states.
type translated struct {
	tp, pp, dp, dcp, pcp int
	ep                   bool
	engine               []string // engine block lines
	launcherOnly         []string
}

// launcherOnlyFlags are the arguments that start, bind or front-end a server and state nothing the
// simulated pool needs beyond what the scenario's pool block already holds: they are recorded for
// the fields.json check and otherwise carry no setting into the scenario.
var launcherOnlyFlags = map[string]bool{
	"--tokenizer-mode": true, "--config-format": true, "--load-format": true,
	"--reasoning-parser": true, "--tool-call-parser": true, "--enable-auto-tool-choice": true,
	"--port": true, "--no-numa-bind": true, "--headless": true,
	"--nnodes": true, "--node-rank": true, "--master-addr": true,
	"--data-parallel-size-local": true, "--data-parallel-start-rank": true, "--data-parallel-address": true,
	"--data-parallel-rpc-port": true, "--data-parallel-hybrid-lb": true,
}

func translate(t *testing.T, c recipeCase, p casePool, argv []string, blockSize int, k *int) translated {
	t.Helper()
	m, order := settle(argv, nil)
	tr := translated{tp: 1, pp: 1, dp: 1, dcp: 1, pcp: 1}
	engine := []string{fmt.Sprintf("block_size: %d", blockSize), "enable_prefix_caching: true", "async_scheduling: true"}
	for _, name := range order {
		v := m[name]
		atoi := func() int {
			n, err := strconv.Atoi(v)
			if err != nil {
				t.Fatalf("%s %s: %q is not an integer", c.name, name, v)
			}
			return n
		}
		switch name {
		case "--tensor-parallel-size":
			tr.tp = atoi()
		case "--pipeline-parallel-size":
			tr.pp = atoi()
		case "--data-parallel-size":
			tr.dp = atoi()
		case "--decode-context-parallel-size":
			tr.dcp = atoi()
		case "--prefill-context-parallel-size":
			tr.pcp = atoi()
		case "--enable-expert-parallel":
			tr.ep = true
		case "--max-model-len":
			engine = append(engine, "max_model_len: "+v)
		case "--max-num-batched-tokens":
			engine = append(engine, "max_num_batched_tokens: "+v)
		case "--max-num-seqs":
			engine = append(engine, "max_num_seqs: "+v)
		case "--cp-kv-cache-interleave-size":
			engine = append(engine, "cp_kv_cache_interleave_size: "+v)
		case "--language-model-only":
			engine = append(engine, "language_model_only: true")
		case "--kernel-config":
			var kc map[string]any
			mustJSON(t, name, v, &kc)
			for key, val := range kc {
				if key != "moe_backend" {
					t.Errorf("unhandled preset argument: %s names %q", name, key)
					continue
				}
				engine = append(engine, fmt.Sprintf("moe_backend: %v", val))
			}
		case "--compilation-config":
			var cc map[string]any
			mustJSON(t, name, v, &cc)
			for key, val := range cc {
				if key != "cudagraph_mode" {
					t.Errorf("unhandled preset argument: %s names %q", name, key)
					continue
				}
				engine = append(engine, fmt.Sprintf("compilation_config.cudagraph_mode: %v", val))
			}
		case "--speculative-config":
			var sc map[string]any
			mustJSON(t, name, v, &sc)
			parts := []string{}
			for _, key := range []string{"method", "num_speculative_tokens", "draft_sample_method"} {
				if val, ok := sc[key]; ok {
					parts = append(parts, fmt.Sprintf("%s: %v", key, val))
					delete(sc, key)
				}
			}
			for key := range sc {
				t.Errorf("unhandled preset argument: %s names %q", name, key)
			}
			engine = append(engine, "speculative_config: {"+strings.Join(parts, ", ")+"}")
			if n, ok := sc["num_speculative_tokens"]; ok {
				_ = n
			}
		case "--kv-transfer-config":
			var kv map[string]any
			mustJSON(t, name, v, &kv)
			if kv["kv_connector"] != "NixlConnector" {
				t.Errorf("unhandled preset argument: %s names connector %v, and only NixlConnector is simulated here", name, kv["kv_connector"])
			}
		default:
			if !launcherOnlyFlags[name] {
				t.Errorf("unhandled preset argument: %s %s (%s on %s)", name, v, c.name, c.hw)
				continue
			}
			tr.launcherOnly = append(tr.launcherOnly, name)
		}
	}
	tr.engine = engine
	return tr
}

func mustJSON(t *testing.T, flag, v string, into any) {
	t.Helper()
	if err := json.Unmarshal([]byte(v), into); err != nil {
		t.Fatalf("%s: %q is not JSON: %v", flag, v, err)
	}
}

// classifiedNames returns the fields.json names a command states: each flag as vLLM names it
// (dashes to underscores, --no-X to X where X is the field), the keys of a JSON-valued flag as
// <flag>.<key>, and each environment variable.
func classifiedNames(argv []string, env map[string]string) (cfg, envs []string) {
	m, order := settle(argv, nil)
	for _, name := range order {
		field := strings.ReplaceAll(strings.TrimPrefix(name, "--"), "-", "_")
		switch name {
		case "--kernel-config", "--compilation-config", "--speculative-config", "--kv-transfer-config":
			var obj map[string]any
			if json.Unmarshal([]byte(m[name]), &obj) == nil {
				for k := range obj {
					cfg = append(cfg, field+"."+k)
				}
				continue
			}
		}
		cfg = append(cfg, field)
	}
	for k := range env {
		envs = append(envs, k)
	}
	return cfg, envs
}

// fieldStatuses is engines/vllm/fields.json by "config:<name>" / "env:<name>".
func fieldStatuses(t *testing.T) map[string]string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "engines", "vllm", "fields.json"))
	if err != nil {
		t.Fatalf("reading the field list: %v", err)
	}
	var doc struct {
		Engine struct {
			Version string `json:"version"`
		} `json:"engine"`
		Fields []struct {
			Name   string `json:"name"`
			Source string `json:"source"`
			Status string `json:"status"`
		} `json:"fields"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("parsing the field list: %v", err)
	}
	if !strings.HasSuffix(doc.Engine.Version, recipesCommit) {
		t.Fatalf("the field list is for %s and the recipes are read at %s: move the constant with the engine", doc.Engine.Version, recipesCommit)
	}
	out := map[string]string{}
	for _, f := range doc.Fields {
		out[f.Source+":"+f.Name] = f.Status
	}
	return out
}

// scenarioFor writes the scenario and deployment of a case under one arm.
func scenarioFor(c recipeCase, hwc struct {
	catalog, fabric string
	perNode, rack   int
	family          string
	sm              string
}, name string, trs []translated, accept string) string {
	var b strings.Builder
	nodes := 0
	for _, p := range c.pools {
		nodes += p.nodes
	}
	fmt.Fprintf(&b, "kind: Scenario\nname: %s\nengine: {kind: vllm}\nmodel: %s\n", name, c.model)
	b.WriteString(accept)
	b.WriteString("coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]\n")
	fmt.Fprintf(&b, "cluster:\n  hardware: %s\n  fabric: %s\n  nodes: %d\n  gpus_per_node: %d\n", hwc.catalog, hwc.fabric, nodes, hwc.perNode)
	if hwc.rack > 0 {
		fmt.Fprintf(&b, "  gpus_per_rack: %d\n", hwc.rack)
	}
	fmt.Fprintf(&b, "---\nkind: Deployment\nname: %s\npools:\n", name)
	for i, p := range c.pools {
		tr := trs[i]
		fmt.Fprintf(&b, "  - role: %s\n    nodes: %d\n    parallel: {tp: %d, pp: %d, dp: %d, dcp: %d, pcp: %d, enable_expert_parallel: %v}\n    engine:\n",
			p.role, p.nodes, tr.tp, tr.pp, tr.dp, tr.dcp, tr.pcp, tr.ep)
		for _, l := range tr.engine {
			fmt.Fprintf(&b, "      %s\n", l)
		}
	}
	if len(c.pools) > 1 {
		b.WriteString("pd_transfer:\n  connector: NixlConnector\nrouter:\n  endpoint_picker_config: router/pd-always.yaml\n")
	}
	return b.String()
}

func TestRecipePresetsRunOnEveryNVIDIAHardware(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs the recipe presets")
	}
	root := recipeSource(t)
	blis := builtBlis(t)
	_, catalog, registry := kernelRepos(t)
	status := fieldStatuses(t)
	routerDir, err := filepath.Abs(filepath.Join("..", "testdata", "scenarios", "router"))
	if err != nil {
		t.Fatal(err)
	}

	recipes := []struct {
		family, rel string
		blockSize   int
	}{
		// block_size is the engine's, not the recipe's: the block-size oracle does not cover these
		// models, so the scenario states it (Mistral: the MLA page of 64 tokens; Inkling: the
		// recurrent layers' alignment, 16).
		{"mistral", "mistralai/Mistral-Large-4-1T-A52B.yaml", 64},
		{"inkling", "thinkingmachines/Inkling-2-Small.yaml", 16},
	}
	const requests = 8
	for _, rc := range recipes {
		r := loadRecipe(t, root, rc.rel)
		cases := buildCases(t, r, rc.family)
		if len(cases) < 8 {
			t.Fatalf("%s: resolved %d commands", rc.family, len(cases))
		}
		for _, c := range cases {
			c := c
			key := fmt.Sprintf("%s/%s/%s/%s", rc.family, c.hw, c.name, c.variant)
			t.Run(key, func(t *testing.T) {
				t.Parallel()
				if _, err := os.Stat(filepath.Join(catalog, "models", c.model, "graph.yaml")); err != nil {
					t.Fatalf("catalog has no model %s: %v", c.model, err)
				}
				var plain map[string]any
				for _, a := range recipeArms {
					if a.spec != "" && (c.noSpec || !r.offers("spec_decoding", c.strategy)) {
						continue
					}
					if a.name == "text-only" && c.textOnly {
						continue // the preset states it in every arm
					}
					var featArgs []string
					if a.feature != nil {
						featArgs = a.feature(r)
					}
					if a.spec != "" {
						featArgs = append(featArgs, r.featureArgs("spec_decoding", a.spec)...)
					}
					if c.textOnly {
						featArgs = append(featArgs, r.featureArgs("text_only", "")...)
					}
					if a.name == "parsers" && rc.family == "inkling" {
						// Inkling's opt-in is tool calling; reasoning is on by default and the arm adds both.
						_ = featArgs
					}
					trs := make([]translated, len(c.pools))
					cfgNames, envNames := map[string]bool{}, map[string]bool{}
					for i, p := range c.pools {
						argv := append(append([]string{}, p.args...), featArgs...)
						trs[i] = translate(t, c, p, argv, rc.blockSize, nil)
						cn, en := classifiedNames(append(argv, p.workerArgs...), p.env)
						for _, n := range cn {
							cfgNames[n] = true
						}
						for _, n := range en {
							envNames[n] = true
						}
					}
					for n := range cfgNames {
						if _, ok := status["config:"+n]; !ok {
							if strings.HasPrefix(n, "no_") {
								if _, ok := status["config:"+strings.TrimPrefix(n, "no_")]; ok {
									continue
								}
							}
							t.Errorf("%s: argument --%s has no status in engines/vllm/fields.json", a.name, strings.ReplaceAll(n, "_", "-"))
						}
					}
					for n := range envNames {
						if _, ok := status["env:"+n]; !ok {
							t.Errorf("%s: environment variable %s has no status in engines/vllm/fields.json", a.name, n)
						}
					}
					accept := ""
					if a.spec != "" {
						ac := recipeAcceptance[rc.family]
						al := ac.greedy
						if a.spec == "dspark_probabilistic" {
							al = ac.probabilistic
						}
						k := 3
						if rc.family == "inkling" {
							k = 7
						}
						accept = fmt.Sprintf("acceptance:\n  value:\n    num_spec_tokens: %d\n    acceptance_length: %g\n    source: %q\n", k, al, ac.source)
					}
					dir := t.TempDir()
					if err := os.CopyFS(filepath.Join(dir, "router"), os.DirFS(routerDir)); err != nil {
						t.Fatal(err)
					}
					file := "case.yaml"
					text := scenarioFor(c, recipeHardware[c.hw], "recipe-"+c.hw+"-"+c.name, trs, accept)
					if err := os.WriteFile(filepath.Join(dir, file), []byte(text), 0o644); err != nil {
						t.Fatal(err)
					}
					args := []string{"run", "--scenarios", dir, "--scenario", file, "--catalog", catalog,
						"--registry", registry, "--rate", "2", "--num-requests", strconv.Itoa(requests),
						"--prompt-tokens", "4096", "--prompt-tokens-min", "4096", "--prompt-tokens-max", "4096",
						"--prompt-tokens-stdev", "0", "--output-tokens", "64", "--output-tokens-min", "64",
						"--output-tokens-max", "64", "--output-tokens-stdev", "0"}
					if len(c.pools) > 1 {
						args = append(args, "--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1")
					}
					cmd := exec.Command(blis, args...)
					var out, errBuf bytes.Buffer
					cmd.Stdout, cmd.Stderr = &out, &errBuf
					runErr := cmd.Run()
					want, refused := expectedRefusals[key]
					armRefused := false
					if w, ok := expectedRefusals[key+"#"+a.name]; ok {
						want, refused, armRefused = w, true, true
					}
					if refused {
						if runErr == nil || !strings.Contains(errBuf.String(), want) {
							t.Errorf("%s: want the refusal %q, got %v\n%s", a.name, want, runErr, lastLines(errBuf.String(), 3))
						}
						if armRefused {
							continue
						}
						break
					}
					if runErr != nil {
						t.Errorf("%s: did not complete: %v\n%s\n%s", a.name, runErr, lastLines(errBuf.String(), 3), text)
						continue
					}
					m := clusterFigures(t, out.String())
					if got := m["completed_requests"]; got != requests {
						t.Errorf("%s: completed %v of %d requests", a.name, got, requests)
					}
					if got := m["dropped_unservable"]; got != 0 {
						t.Errorf("%s: dropped %v requests as unservable", a.name, got)
					}
					doc := resultDoc(t, out.String())
					coefs, _ := resultValue(doc, "selected_entries.coefficients")
					list, _ := coefs.([]any)
					if len(list) == 0 {
						t.Errorf("%s: the run selected no coefficients", a.name)
					}
					for _, e := range list {
						em, _ := e.(map[string]any)
						if method, _ := em["method"].(string); method == "" {
							t.Errorf("%s: coefficient %v states no method", a.name, em["name"])
						}
					}
					metrics, _ := resultValue(doc, "metrics")
					if a.name == "plain" {
						plain, _ = metrics.(map[string]any)
					} else if a.checksum && plain != nil && !reflect.DeepEqual(plain, metrics) {
						t.Errorf("%s: the front-end-only parsers changed the metrics", a.name)
					} else if a.spec != "" && plain != nil && reflect.DeepEqual(plain, metrics) {
						t.Errorf("%s: speculative decoding priced nothing: the metrics equal the plain run's", a.name)
					}
				}
			})
		}
	}
}
