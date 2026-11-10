// Original to blis-vllm.

package engineconfig

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// The engine block takes vLLM's own setting names (design X1: what a user hands vLLM is what they
// hand BLIS), so a name in fields.json is a name in the block. Engine's Go fields keep the shape
// the engine reads them in, and their yaml tags name them in that shape; FromVLLM translates a
// block written in vLLM's names into it, and ToVLLM writes it back, so Decode and Encode are the
// one place the two meet. A block written in the earlier names is refused, with the new one named.
//
// vLLM's EngineArgs takes a nested config object under the name of its class
// (compilation_config, attention_config, speculative_config, eplb_config, kv_events_config,
// engram_config) and the rest flat. A setting's vLLM name is the dotted path used in fields.json.

// move is a block key: where the Go-shaped block holds it, where vLLM's names put it.
type move struct {
	internal string // top-level key, or "group.key" for a key inside a nested struct
	external string // top-level key, or "group.key"
}

var moves = []move{
	{"cudagraph_mode", "compilation_config.cudagraph_mode"},
	{"decoder_replay_cudagraph_capture_sizes", "compilation_config.decoder_replay_cudagraph_capture_sizes"},
	{"custom_ops", "compilation_config.custom_ops"},
	{"use_trtllm_attention", "attention_config.use_trtllm_attention"},
	{"mla_prefill_backend", "attention_config.mla_prefill_backend"},
	{"indexer_kv_dtype", "attention_config.indexer_kv_dtype"},
	{"cache_dtype", "kv_cache_dtype"},
	{"dbo.enable_dbo", "enable_dbo"},
	{"dbo.dbo_decode_token_threshold", "dbo_decode_token_threshold"},
	{"dbo.dbo_prefill_token_threshold", "dbo_prefill_token_threshold"},
	{"eplb.enable_eplb", "enable_eplb"},
	{"eplb.num_redundant_experts", "eplb_config.num_redundant_experts"},
	{"eplb.window_size", "eplb_config.window_size"},
	{"eplb.step_interval", "eplb_config.step_interval"},
	{"kv_events.enable_kv_cache_events", "kv_events_config.enable_kv_cache_events"},
	{"kv_events.publisher", "kv_events_config.publisher"},
	{"kv_events.topic", "kv_events_config.topic"},
	{"speculative.method", "speculative_config.method"},
	{"speculative.num_spec_tokens", "speculative_config.num_speculative_tokens"},
	{"speculative.draft_model", "speculative_config.model"},
	{"speculative.kv_cache_dtype", "speculative_config.kv_cache_dtype"},
	{"speculative.draft_sample_method", "speculative_config.draft_sample_method"},
	{"speculative.rejection_sample_method", "speculative_config.rejection_sample_method"},
	{"speculative.enable_adaptive_verification", "speculative_config.enable_adaptive_verification"},
}

// rustFrontendEnv is the environment variable vLLM selects its front end with
// (vllm/envs.py:170@5df63a72); the block's earlier frontend key was a name for it.
const rustFrontendEnv = "VLLM_USE_RUST_FRONTEND"

// refusedKeys are earlier top-level names that now have another spelling, with what to write.
var refusedKeys = map[string]string{
	"frontend": "is vLLM's environment variable " + rustFrontendEnv + ": write env: {" + rustFrontendEnv + ": \"1\"}",
}

// FromVLLM translates an engine block written in vLLM's setting names into the Go-shaped block
// Decode reads. A key written in an earlier name, or a key of a group written at the wrong level,
// is an error naming the key to use. The input is not changed.
func FromVLLM(n *yaml.Node) (*yaml.Node, error) {
	if n == nil || n.Kind != yaml.MappingNode {
		return n, nil
	}
	out := &yaml.Node{Kind: yaml.MappingNode, Tag: n.Tag, Style: n.Style, HeadComment: n.HeadComment,
		LineComment: n.LineComment, FootComment: n.FootComment, Line: n.Line, Column: n.Column}
	internal := map[string]move{}
	external := map[string]move{}
	for _, m := range moves {
		internal[m.internal] = m
		external[m.external] = m
	}
	groups := map[string]*yaml.Node{} // internal group name -> its mapping in out
	add := func(m *yaml.Node, k, v *yaml.Node) { m.Content = append(m.Content, k, v) }
	groupOf := func(name string, at *yaml.Node) *yaml.Node {
		if g, ok := groups[name]; ok {
			return g
		}
		g := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map", Line: at.Line, Column: at.Column}
		groups[name] = g
		add(out, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: name, Line: at.Line, Column: at.Column}, g)
		return g
	}
	seen := map[string]bool{}
	for i := 0; i+1 < len(n.Content); i += 2 {
		k, v := n.Content[i], n.Content[i+1]
		if k.Kind != yaml.ScalarNode {
			add(out, k, v)
			continue
		}
		key := k.Value
		if why, ok := refusedKeys[key]; ok {
			return nil, fmt.Errorf("line %d: engine key %q %s", k.Line, key, why)
		}
		// A key of the earlier layout: refused with vLLM's name.
		if m, ok := internal[key]; ok && !strings.Contains(m.internal, ".") {
			return nil, fmt.Errorf("line %d: engine key %q is now %q, vLLM's own name", k.Line, key, m.external)
		}
		switch key {
		case "dbo", "eplb", "kv_events", "speculative":
			return nil, fmt.Errorf("line %d: engine key %q is now %s, vLLM's own names (%s)", k.Line, key, groupTarget(key), groupNames(key))
		}
		if _, dup := seen[key]; dup {
			return nil, fmt.Errorf("line %d: engine key %q is stated twice", k.Line, key)
		}
		seen[key] = true
		switch key {
		case "compilation_config", "attention_config", "speculative_config", "eplb_config", "kv_events_config":
			if v.Kind != yaml.MappingNode {
				add(out, k, v)
				continue
			}
			for j := 0; j+1 < len(v.Content); j += 2 {
				ik, iv := v.Content[j], v.Content[j+1]
				path := key + "." + ik.Value
				m, ok := external[path]
				if !ok {
					// Not a key the block knows: kept under the Go-shaped group so the strict
					// decoder names it.
					add(groupOf(goGroup(key), ik), ik, iv)
					continue
				}
				ig, leaf, _ := strings.Cut(m.internal, ".")
				if leaf == "" {
					add(out, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: ig, Line: ik.Line, Column: ik.Column}, iv)
					continue
				}
				add(groupOf(ig, ik), &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: leaf, Line: ik.Line, Column: ik.Column}, iv)
			}
		case "env":
			add(out, k, v)
		default:
			if m, ok := external[key]; ok {
				ig, leaf, _ := strings.Cut(m.internal, ".")
				if leaf == "" {
					add(out, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: ig, Line: k.Line, Column: k.Column}, v)
				} else {
					add(groupOf(ig, k), &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: leaf, Line: k.Line, Column: k.Column}, v)
				}
				continue
			}
			add(out, k, v)
		}
	}
	// env: {VLLM_USE_RUST_FRONTEND: ...} selects the front end; the Go-shaped block holds it as
	// frontend, and the variable leaves env so it is not read twice.
	for i := 0; i+1 < len(out.Content); i += 2 {
		if out.Content[i].Value != "env" || out.Content[i+1].Kind != yaml.MappingNode {
			continue
		}
		env := out.Content[i+1]
		var kept []*yaml.Node
		for j := 0; j+1 < len(env.Content); j += 2 {
			if env.Content[j].Value != rustFrontendEnv {
				kept = append(kept, env.Content[j], env.Content[j+1])
				continue
			}
			fe := "python"
			switch strings.ToLower(env.Content[j+1].Value) {
			case "1", "true":
				fe = "rust"
			case "0", "false", "":
			default:
				return nil, fmt.Errorf("line %d: %s is %q; vLLM reads it as 0 or 1", env.Content[j].Line, rustFrontendEnv, env.Content[j+1].Value)
			}
			add(out, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "frontend"},
				&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: fe})
		}
		env.Content = kept
		if len(kept) == 0 {
			out.Content = append(out.Content[:i], out.Content[i+2:]...)
		}
		break
	}
	return out, nil
}

// goGroup is the Go-shaped group a vLLM config object's keys land in.
func goGroup(external string) string {
	switch external {
	case "speculative_config":
		return "speculative"
	case "eplb_config":
		return "eplb"
	case "kv_events_config":
		return "kv_events"
	}
	return external
}

// groupTarget is how the earlier group is written in vLLM's names: a config object, or flat keys.
func groupTarget(old string) string {
	switch old {
	case "dbo":
		return "flat keys"
	case "eplb":
		return "enable_eplb and eplb_config"
	case "kv_events":
		return "kv_events_config"
	}
	return "speculative_config"
}

func groupNames(old string) string {
	var out []string
	for _, m := range moves {
		if strings.HasPrefix(m.internal, old+".") {
			out = append(out, m.external)
		}
	}
	if len(out) > 5 {
		out = append(out[:4], "...")
	}
	return strings.Join(out, ", ")
}

// ToVLLM writes a Go-shaped block in vLLM's setting names. Keys keep their order of first
// appearance; a group's keys are gathered under its vLLM config object.
func ToVLLM(n *yaml.Node) *yaml.Node {
	if n == nil || n.Kind != yaml.MappingNode {
		return n
	}
	byInternal := map[string]move{}
	for _, m := range moves {
		byInternal[m.internal] = m
	}
	out := &yaml.Node{Kind: yaml.MappingNode, Tag: n.Tag, Style: n.Style}
	groups := map[string]*yaml.Node{}
	str := func(s string) *yaml.Node { return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: s} }
	put := func(path string, v *yaml.Node) {
		g, leaf, nested := strings.Cut(path, ".")
		if !nested {
			out.Content = append(out.Content, str(path), v)
			return
		}
		m, ok := groups[g]
		if !ok {
			m = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			groups[g] = m
			out.Content = append(out.Content, str(g), m)
		}
		m.Content = append(m.Content, str(leaf), v)
	}
	var env *yaml.Node
	frontend := ""
	for i := 0; i+1 < len(n.Content); i += 2 {
		k, v := n.Content[i], n.Content[i+1]
		switch k.Value {
		case "dbo", "eplb", "kv_events", "speculative":
			if v.Kind != yaml.MappingNode {
				continue
			}
			for j := 0; j+1 < len(v.Content); j += 2 {
				if m, ok := byInternal[k.Value+"."+v.Content[j].Value]; ok {
					put(m.external, v.Content[j+1])
				}
			}
		case "frontend":
			frontend = v.Value
		case "env":
			env = v
		default:
			if m, ok := byInternal[k.Value]; ok {
				put(m.external, v)
			} else {
				out.Content = append(out.Content, k, v)
			}
		}
	}
	if frontend == "rust" || frontend == "python" || env != nil {
		e := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		if env != nil {
			e.Content = append(e.Content, env.Content...)
		}
		switch frontend {
		case "rust":
			e.Content = append(e.Content, str(rustFrontendEnv), &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "1"})
		case "python":
			e.Content = append(e.Content, str(rustFrontendEnv), &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "0"})
		}
		out.Content = append(out.Content, str("env"), e)
	}
	return out
}

// VLLMKeys is every key an engine block takes in vLLM's names: the top-level EngineArgs names
// (a config object such as compilation_config is one) and, for each config object, the keys it
// takes, as the dotted names fields.json uses. It is what the block accepts, derived from Engine
// and the translation, so a test can hold it against vLLM's own EngineArgs.
func VLLMKeys() (top []string, nested map[string][]string) {
	nested = map[string][]string{}
	moved := map[string]bool{}
	for _, m := range moves {
		moved[m.internal] = true
		if g, leaf, ok := strings.Cut(m.external, "."); ok {
			nested[g] = append(nested[g], leaf)
		} else {
			top = append(top, m.external)
		}
	}
	groupsGone := map[string]bool{"dbo": true, "eplb": true, "kv_events": true, "speculative": true, "frontend": true}
	t := reflect.TypeOf(Engine{})
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("yaml"), ",")
		if name == "" || name == "-" || moved[name] || groupsGone[name] {
			continue
		}
		top = append(top, name)
	}
	for g := range nested {
		top = append(top, g)
		sort.Strings(nested[g])
	}
	sort.Strings(top)
	return slicesCompact(top), nested
}

func slicesCompact(s []string) []string {
	out := s[:0]
	for i, v := range s {
		if i == 0 || v != s[i-1] {
			out = append(out, v)
		}
	}
	return out
}
