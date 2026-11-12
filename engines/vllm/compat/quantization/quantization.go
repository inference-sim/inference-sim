// Original to blis-vllm.

// Package quantization lowers the `quantization` a deployment states to the weight layout the
// engine serves, for the catalog checkpoints that state their own quantization recipe.
//
// vLLM does not serve what `--quantization` names. ModelConfig._verify_quantization
// (vllm/config/model.py:1370-1449@5df63a72) reads the checkpoint's quantization_config, asks every
// quantization class whether it claims the checkpoint (override_quantization_method,
// vllm/model_executor/layers/quantization/modelopt.py:449-455, 772-778, 1205-1211,
// 1627-1633@5df63a72), and then either takes the claimed method whatever was stated (:1416-1437),
// accepts a stated value equal to the checkpoint's method, or raises (:1440-1448). The method it
// settles on builds each layer's weights per the checkpoint's recipe: for a ModelOpt mixed-precision
// checkpoint, per layer, from the `quantized_layers` mapping
// (ModelOptMixedPrecisionConfig._resolve_quant_algo, modelopt.py:1692-1790@5df63a72).
//
// The kernel prices a layer by the dtype its weights are stored in, and its catalog graph records
// that dtype for each class of layers. The engine model therefore states the layout vLLM serves
// (enginesettings.ServedWeights) and the kernel proves the graph stores exactly it, refusing
// otherwise. This package computes that statement. It reads no file at run time: the answers are
// oracle/quantization/oracle.py's, which runs vLLM's own ModelConfig and quantization configs over
// each checkpoint's config.json and is embedded here as resolved.json; the checkpoint is named by
// the digest of that config.json, which its catalog graph records.
package quantization

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
)

// Upstream is the vLLM commit the recorded answers are for.
const Upstream = "5df63a7203cdb9328d824a27b682e77c7b9928be"

// unlisted is the stated value oracle.py uses for "a string no vLLM release names"; vLLM's answer to
// it is its answer to every value the oracle did not list.
const unlisted = "not_a_quantization_method"

// unquantized is oracle.py's word for a layer the quantization config leaves in the model dtype.
const unquantized = "unquantized"

//go:embed resolved.json
var recorded []byte

type table struct {
	VLLM        string                `json:"vllm"`
	Checkpoints map[string]checkpoint `json:"checkpoints"`
}

type checkpoint struct {
	ConfigSHA256     string `json:"config_sha256"`
	ModelType        string `json:"model_type"`
	CheckpointMethod string `json:"checkpoint_quant_method"`
	// OwnMethod is what vLLM resolves when nothing is stated: the method the checkpoint's recipe
	// is served by.
	OwnMethod string          `json:"resolved_without_quantization"`
	Cases     []row           `json:"cases"`
	Control   json.RawMessage `json:"control_without_quantization_config"`
	Layout    *layout         `json:"layout"`
}

type row struct {
	Stated   *string `json:"stated"`
	Resolved *string `json:"resolved"`
	Error    *string `json:"error"`
}

type layout struct {
	ConfigClass      string                    `json:"config_class"`
	MXFP8BlockSize   int                       `json:"mxfp8_block_size"`
	UnquantizedDType string                    `json:"unquantized_dtype"`
	Classes          map[string]map[string]int `json:"classes"`
}

func load() (map[string]checkpoint, error) { return parse(recorded) }

// parse reads the recorded answers, keyed by the digest of each checkpoint's config.json.
func parse(raw []byte) (map[string]checkpoint, error) {
	var t table
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&t); err != nil {
		return nil, fmt.Errorf("quantization: the embedded resolved.json is malformed: %w", err)
	}
	if t.VLLM != Upstream {
		return nil, fmt.Errorf("quantization: resolved.json is for vLLM %s, this build models %s", t.VLLM, Upstream)
	}
	bySHA := map[string]checkpoint{}
	for name, c := range t.Checkpoints {
		if _, dup := bySHA[c.ConfigSHA256]; dup {
			return nil, fmt.Errorf("quantization: two checkpoints (%s among them) record config %s", name, c.ConfigSHA256)
		}
		bySHA[c.ConfigSHA256] = c
	}
	return bySHA, nil
}

// Lowered is the engine's statement of what vLLM serves for a stated quantization.
type Lowered struct {
	// Method is the quantization method vLLM settles on, the checkpoint's own.
	Method string
	// Served is the layout vLLM serves the checkpoint in, class by class.
	Served enginesettings.ServedWeights
	// Requested is the stated value when vLLM replaces it with Method (the checkpoint's recipe is
	// claimed by another class, :1416-1437), and empty when the value was unset or already Method.
	Requested string
}

// Refusal is a stated quantization this engine model refuses: vLLM raises on it, or it resolves to
// a method whose layout the kernel does not price.
type Refusal struct {
	Stated  string
	Message string
}

func (r *Refusal) Error() string { return r.Message }

// Lower resolves the quantization stated for the checkpoint whose config.json has the given SHA-256.
//
// It returns nil, nil for a checkpoint the recorded answers give no layout for (every checkpoint but
// the MiniMax-M3 ones): the caller keeps the setting as stated, and the kernel prices it as before.
// For one it covers, a value vLLM refuses returns a *Refusal carrying vLLM's own message, and a value
// that resolves to the checkpoint's own method (or that its recipe's class claims) returns the layout.
func Lower(configSHA256, stated string) (*Lowered, error) {
	by, err := load()
	if err != nil {
		return nil, err
	}
	return lower(by, configSHA256, stated)
}

func lower(by map[string]checkpoint, configSHA256, stated string) (*Lowered, error) {
	cp, ok := by[configSHA256]
	if !ok || cp.Layout == nil {
		return nil, nil
	}
	r, err := cp.rowFor(stated)
	if err != nil {
		return nil, err
	}
	if r.Error != nil {
		msg := *r.Error
		if r.Stated != nil && *r.Stated == unlisted {
			msg = strings.ReplaceAll(msg, unlisted, stated)
		}
		return nil, &Refusal{Stated: stated, Message: fmt.Sprintf(
			"quantization %q is refused: vLLM v0.32.0 raises, ModelConfig._verify_quantization "+
				"(vllm/config/model.py:1440-1448@5df63a72): %s", stated, msg)}
	}
	method := *r.Resolved
	if method != cp.OwnMethod {
		return nil, &Refusal{Stated: stated, Message: fmt.Sprintf(
			"quantization %q resolves to %q in vLLM v0.32.0, not the checkpoint's own %q; this engine "+
				"model states no served layout for it, so the kernel cannot price it",
			stated, method, cp.OwnMethod)}
	}
	served, err := cp.Layout.served()
	if err != nil {
		return nil, &Refusal{Stated: stated, Message: fmt.Sprintf("quantization %q: %v", stated, err)}
	}
	out := &Lowered{Method: method, Served: served}
	if stated != "" && stated != method {
		out.Requested = stated
	}
	return out, nil
}

// rowFor is the recorded answer for a stated value: its own row, or the row of a value vLLM does not
// name (every such value is answered alike).
func (c checkpoint) rowFor(stated string) (row, error) {
	var other *row
	for i, r := range c.Cases {
		switch {
		case r.Stated == nil && stated == "":
			return r, nil
		case r.Stated == nil:
		case *r.Stated == stated:
			return r, nil
		case *r.Stated == unlisted:
			other = &c.Cases[i]
		}
	}
	if other == nil || stated == "" {
		return row{}, fmt.Errorf("quantization: the recorded answers for config %s have no row for %q", c.ConfigSHA256, stated)
	}
	return *other, nil
}

// served maps the algorithm each class of layers runs to the catalog's dtype words. The kernel prices
// one dtype per class, so a class vLLM serves at more than one (a mixed-precision recipe that splits
// the layers of one class) is refused rather than priced at one of them.
//
// The router and the token embedding are not read from the recipe, because it does not govern them:
// MiniMax-M3's router is a GateLinear built with no quantization config and float32 weights
// (vllm/models/minimax_m3/nvidia/model.py:231-238@5df63a72), and no ModelOpt config quantizes the token
// embedding (get_quant_method returns None for it, modelopt.py:244-257, 1810-1860@5df63a72), so it stays
// in the model dtype.
func (l layout) served() (enginesettings.ServedWeights, error) {
	model, err := dtypeWord(l.UnquantizedDType)
	if err != nil {
		return enginesettings.ServedWeights{}, err
	}
	var sw enginesettings.ServedWeights
	for _, c := range []struct {
		class string
		dtype *string
		scale *int
	}{
		{"linear", &sw.Linear, &sw.LinearScaleElements},
		{"routed_experts", &sw.RoutedExperts, nil},
		{"head", &sw.Head, nil},
	} {
		algos := l.Classes[c.class]
		if len(algos) != 1 {
			names := make([]string, 0, len(algos))
			for a := range algos {
				names = append(names, a)
			}
			sort.Strings(names)
			return sw, fmt.Errorf("the checkpoint's recipe serves its %s layers in %d weight formats (%s); the kernel "+
				"prices one format per class of layers", c.class, len(algos), strings.Join(names, ", "))
		}
		for a := range algos {
			d, scale, err := l.format(a, model)
			if err != nil {
				return sw, fmt.Errorf("%s layers: %w", c.class, err)
			}
			*c.dtype = d
			if c.scale != nil {
				*c.scale = scale
			}
		}
	}
	sw.Router = "fp32"
	sw.Embedding = model
	return sw, nil
}

// format is the dtype word of a ModelOpt algorithm and the number of weights that share one scale
// byte where the format keeps a block scale (0 for none): MXFP8 is e4m3 with one e8m0 scale per 32
// (MXFP8_BLOCK_SIZE, vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:11@5df63a72).
func (l layout) format(algo, modelDType string) (string, int, error) {
	switch algo {
	case "MXFP8":
		return "fp8", l.MXFP8BlockSize, nil
	case "NVFP4", "W4A16_NVFP4":
		return "nvfp4", 0, nil
	case "FP8":
		return "fp8", 0, nil
	case unquantized:
		return modelDType, 0, nil
	}
	return "", 0, fmt.Errorf("the algorithm %q has no weight width the kernel prices", algo)
}

func dtypeWord(torch string) (string, error) {
	switch torch {
	case "bfloat16":
		return "bf16", nil
	case "float16":
		return "fp16", nil
	}
	return "", fmt.Errorf("the checkpoint's model dtype %q has no catalog word", torch)
}

// Checkpoints lists the config digests this package lowers (those with a recorded layout), by model
// name, for the test that holds them to the pinned catalog.
func Checkpoints() (map[string]string, error) {
	var t table
	if err := json.Unmarshal(recorded, &t); err != nil {
		return nil, err
	}
	out := map[string]string{}
	for name, c := range t.Checkpoints {
		if c.Layout != nil {
			out[name] = c.ConfigSHA256
		}
	}
	return out, nil
}
